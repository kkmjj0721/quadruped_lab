# Copyright (c) 2026-2027 zh
"""
内容：
    按原锁文件测速 Python、PyPI 与 Torch 下载源，使用临时下载副本同步环境。

用法：
    bash setup.sh
    python3 tools/setup_env.py --torch-backend cu130 --probe-only
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

from setup_downloads import (
    DownloadSource,
    PYPI_MIRRORS,
    TORCH_MIRRORS,
    ensure_python,
    normalize_root,
    print_probes,
    probe_sources,
    python_download_sources,
    safe_url,
)


# ---------- 锁文件与下载副本 ----------
def read_toml(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


def select_wheel(lock: dict, name: str, backend: str | None = None) -> dict:
    for package in lock["package"]:
        if package["name"] != name or (backend and not package["version"].endswith("+" + backend)):
            continue
        for wheel in package.get("wheels", []):
            filename = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
            if "cp311-cp311-" in filename and "linux" in filename and "x86_64" in filename:
                if not wheel.get("hash", "").startswith("sha256:"):
                    raise RuntimeError(f"Missing SHA256 for {filename}.")
                return wheel
    raise RuntimeError(f"No locked Linux x86_64 CPython 3.11 wheel for {name} ({backend or 'default'}).")


def candidates(roots: list[str] | None, defaults: tuple) -> list[tuple[str, str]]:
    values = [(f"custom-{index}", value) for index, value in enumerate(roots, 1)] if roots else defaults
    return [(name, normalize_root(root)) for name, root in values]


def package_sources(lock: dict, backend: str, pypi_roots: list[str] | None, torch_roots: list[str] | None):
    numpy_url = select_wheel(lock, "numpy")["url"]
    if "/packages/" not in numpy_url:
        raise RuntimeError("The locked PyPI sample does not use a supported /packages/ path.")
    pypi_suffix = numpy_url.split("/packages/", 1)[1]
    pypi = [DownloadSource(name, root, (root + "/" + pypi_suffix,)) for name, root in candidates(pypi_roots, PYPI_MIRRORS)]
    filenames = [urlsplit(select_wheel(lock, name, backend)["url"]).path.rsplit("/", 1)[-1] for name in ("torch", "torchvision")]
    torch = [
        DownloadSource(name, root, tuple(root + "/" + backend + "/" + filename for filename in filenames))
        for name, root in candidates(torch_roots, TORCH_MIRRORS)
    ]
    return pypi, torch


def artifact_metadata(lock: dict) -> dict:
    result = copy.deepcopy(lock)
    for package in result["package"]:
        artifacts = list(package.get("wheels", []))
        if "sdist" in package:
            artifacts.append(package["sdist"])
        for artifact in artifacts:
            artifact.pop("url", None)
    return result


def rewrite_lock(text: str, backend: str, pypi_root: str | None, torch_root: str | None) -> str:
    lock = read_toml(text)
    for package in lock["package"]:
        artifacts = list(package.get("wheels", []))
        if "sdist" in package:
            artifacts.append(package["sdist"])
        for artifact in artifacts:
            original = artifact.get("url", "")
            replacement = original
            if package["name"] in {"torch", "torchvision"}:
                if torch_root and package["version"].endswith("+" + backend):
                    replacement = torch_root + "/" + backend + "/" + urlsplit(original).path.rsplit("/", 1)[-1]
            elif pypi_root and "/packages/" in original:
                replacement = pypi_root + "/" + original.split("/packages/", 1)[1]
            if replacement != original:
                if unquote(urlsplit(original).path.rsplit("/", 1)[-1]) != unquote(urlsplit(replacement).path.rsplit("/", 1)[-1]):
                    raise RuntimeError("A mirror changed the locked artifact filename.")
                # 仅替换完整 URL 字段，不改动版本、依赖、registry 或校验信息。
                text = text.replace('url = "' + original + '"', 'url = "' + replacement + '"')
    if artifact_metadata(lock) != artifact_metadata(read_toml(text)):
        raise RuntimeError("The download overlay changed locked metadata.")
    return text


def can_overlay(manifest: dict, lock: dict, sync_args: list[str]) -> bool:
    uv = manifest.get("tool", {}).get("uv", {})
    if uv.get("package") is not False or "workspace" in uv:
        return False
    for package in lock["package"]:
        source = package.get("source", {})
        if source != {"virtual": "."} and set(source) != {"registry"}:
            return False
    # 这些参数改变解析或相对路径上下文，沿用原项目的 uv 行为。
    contextual = {
        "--find-links", "-f", "--default-index", "--index", "--index-url", "--extra-index-url", "--no-index",
        "--no-sources", "--no-sources-package", "--upgrade", "-U", "--upgrade-package", "-P", "--upgrade-group",
        "--resolution", "--prerelease", "--prerelease-package", "--exclude-newer", "--exclude-newer-package",
        "--fork-strategy", "--index-strategy", "--locked",
    }
    return not any(arg.split("=", 1)[0] in contextual or (arg.startswith(("-f", "-P")) and not arg.startswith("--")) for arg in sync_args)


def validate_sync_args(sync_args: list[str]) -> None:
    protected = {
        "--extra", "--all-extras", "--no-extra", "--active", "--project", "--directory", "--python", "--script",
        "--package", "--all-packages", "--config-file", "--no-config", "--frozen", "--no-locked", "--no-frozen",
    }
    for argument in sync_args:
        if argument.split("=", 1)[0] in protected or (argument.startswith("-p") and not argument.startswith("--")):
            raise ValueError(f"Cannot forward {argument}: the helper controls the project, backend, Python, and lock verification.")


# ---------- 安装进程 ----------
def run_sync(command: list[str], *, cwd: Path, env: dict, timeout: float) -> int:
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # 即使 uv 已退出，也终止仍留在独立进程组内的辅助进程。
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(f"Download attempt exceeded {timeout:g} seconds; trying the next source.", flush=True)
        return 124


def direct_command(uv: str, project: Path, python_version: str, backend: str, sync_args: list[str]) -> list[str]:
    return [uv, "sync", "--project", str(project), "--python", python_version, "--locked", "--extra", backend, *sync_args]


def attempt_pairs(pypi: list, torch: list, *, official_fallback: bool) -> list[tuple]:
    pairs = []
    for diagonal in range(len(pypi) + len(torch) - 1):
        for index in range(len(pypi)):
            other = diagonal - index
            if 0 <= other < len(torch):
                pairs.append((pypi[index].source.root, torch[other].source.root))
    pairs = pairs[:5]
    if official_fallback:
        pairs.append((None, None))
    return list(dict.fromkeys(pairs))


def synchronize(project: Path, args, python_version: str, lock_text: str, pypi: list, torch: list) -> int:
    original_hash = hashlib.sha256(lock_text.encode()).hexdigest()
    base = project / ".artifacts" / "setup-downloads"
    base.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    pairs = attempt_pairs(pypi, torch, official_fallback=not args.pypi_mirror and not args.torch_mirror)
    if not pairs:
        raise RuntimeError("No usable download sources; check the network or use --skip-probes.")
    code = 1
    try:
        for attempt, (pypi_root, torch_root) in enumerate(pairs, 1):
            print(f"Synchronizing with download source combination {attempt}/{len(pairs)}.", flush=True)
            print(f"PyPI downloads: {safe_url(pypi_root) if pypi_root else 'original locked URLs'}; Torch downloads: {safe_url(torch_root) if torch_root else 'original locked URLs'}.", flush=True)
            with tempfile.TemporaryDirectory(prefix="attempt-", dir=base) as directory:
                overlay = Path(directory)
                for filename in ("pyproject.toml", ".python-version", "README.md"):
                    source = project / filename
                    if source.exists():
                        shutil.copy2(source, overlay / filename)
                (overlay / "uv.lock").write_text(rewrite_lock(lock_text, args.torch_backend, pypi_root, torch_root))
                command = [args.uv, "sync", "--project", str(overlay), "--python", python_version, "--frozen", "--extra", args.torch_backend, *args.sync_args]
                code = run_sync(command, cwd=project, env=env, timeout=args.download_timeout)
            if code == 0:
                break
    finally:
        if hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest() != original_hash:
            raise RuntimeError("The original uv.lock changed during installation; it was not overwritten by the helper.")
    return code


# ---------- 参数与主入口 ----------
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Probe real download artifacts and synchronize the locked environment.")
    result.add_argument("--uv", default="uv", help=argparse.SUPPRESS)
    result.add_argument("--torch-backend", choices=("cpu", "cu130"), required=True)
    result.add_argument("--skip-probes", action="store_true", help="Use the original locked uv sync without probing.")
    result.add_argument("--probe-only", action="store_true", help="Probe download channels without synchronizing project dependencies.")
    result.add_argument("--python-mirror", action="append", help="Python standalone mirror root (repeatable).")
    result.add_argument("--pypi-mirror", action="append", help="PyPI artifact root including /packages (repeatable).")
    result.add_argument("--torch-mirror", action="append", help="Torch artifact root preceding /cpu or /cu130 (repeatable).")
    result.add_argument("--probe-timeout", type=float, default=8.0, help="Per-artifact probe timeout in seconds (default: 8).")
    result.add_argument("--probe-bytes", type=int, default=524288, help="Maximum bytes read from each probe artifact (default: 524288).")
    result.add_argument("--download-timeout", type=float, default=1800.0, help="Maximum seconds for each synchronization attempt (default: 1800).")
    result.add_argument("--python-ready", help=argparse.SUPPRESS)
    result.add_argument("sync_args", nargs=argparse.REMAINDER, help="uv sync arguments after --.")
    return result


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.sync_args[:1] == ["--"]:
        args.sync_args = args.sync_args[1:]
    try:
        validate_sync_args(args.sync_args)
    except ValueError as exc:
        argument_parser.error(str(exc))
    if not all(math.isfinite(value) and value > 0 for value in (args.probe_timeout, args.download_timeout)) or not 4 <= args.probe_bytes <= 8 * 1024 * 1024:
        argument_parser.error("Timeouts must be positive; --probe-bytes must be between 4 and 8388608.")
    if args.skip_probes and args.probe_only:
        argument_parser.error("--skip-probes and --probe-only cannot be combined.")
    project = Path(__file__).resolve().parents[1]
    python_version = (project / ".python-version").read_text().split("#", 1)[0].strip()
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    command = direct_command(args.uv, project, python_version, args.torch_backend, args.sync_args)
    offline = os.environ.get("UV_OFFLINE", "").lower() in {"1", "true"} or "--offline" in args.sync_args
    if offline and args.probe_only:
        argument_parser.error("--probe-only cannot run in offline mode.")
    if args.skip_probes or offline or any(flag in args.sync_args for flag in ("--check", "--dry-run")):
        print("Using the original locked synchronization without network probes.", flush=True)
        return run_sync(command, cwd=project, env=env, timeout=args.download_timeout)
    if not args.probe_only and not args.sync_args:
        installed = subprocess.run([*command, "--check", "--offline"], cwd=project, env=env, capture_output=True)
        if installed.returncode == 0:
            print("The environment already matches uv.lock; skipping download probes.", flush=True)
            return 0
    python_path = args.python_ready or ensure_python(
        args.uv, python_version, mirrors=args.python_mirror, offline=args.probe_only, timeout=args.probe_timeout,
        probe_bytes=args.probe_bytes, install_timeout=min(args.download_timeout, 600),
    )
    if Path(sys.executable).resolve() != Path(python_path).resolve():
        os.execv(python_path, [python_path, "-B", str(Path(__file__).resolve()), "--python-ready", python_path, *sys.argv[1:]])
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The download helper requires the project's Python 3.11 interpreter.")
    lock_text = (project / "uv.lock").read_bytes().decode("utf-8")
    manifest = read_toml((project / "pyproject.toml").read_text())
    lock = read_toml(lock_text)
    if not can_overlay(manifest, lock, args.sync_args):
        if args.probe_only:
            raise RuntimeError("This project or these arguments cannot use a download overlay.")
        print("Project packaging or path/index arguments require original locked synchronization.", flush=True)
        return run_sync(command, cwd=project, env=env, timeout=args.download_timeout)
    check = subprocess.run([args.uv, "lock", "--project", str(project), "--check", "--python", python_version], cwd=project, env=env)
    if check.returncode:
        return check.returncode
    if args.probe_only:
        python_results = probe_sources(python_download_sources(args.uv, python_version, args.python_mirror), timeout=args.probe_timeout, probe_bytes=args.probe_bytes)
        print_probes("Python archive channels", python_results)
    pypi_sources, torch_sources = package_sources(lock, args.torch_backend, args.pypi_mirror, args.torch_mirror)
    pypi_results = probe_sources(pypi_sources, timeout=args.probe_timeout, probe_bytes=args.probe_bytes)
    print_probes("PyPI wheel channels", pypi_results)
    torch_results = probe_sources(torch_sources, timeout=args.probe_timeout, probe_bytes=args.probe_bytes)
    print_probes("Torch and torchvision wheel channels", torch_results)
    pypi = [result for result in pypi_results if result.available]
    torch = [result for result in torch_results if result.available]
    if args.probe_only:
        if not pypi or not torch or not any(result.available for result in python_results):
            raise RuntimeError("At least one download category has no available source.")
        print("Download probing completed; project dependencies were not synchronized.", flush=True)
        return 0
    return synchronize(project, args, python_version, lock_text, pypi, torch)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Installation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
