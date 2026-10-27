# Copyright (c) 2026-2027 zh
"""
内容：
    使用标准库测速实际安装文件的下载通道，并准备固定版本的 uv Python。
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------- 下载源与日志 ----------

PYTHON_MIRRORS = (
    ("Astral", "https://releases.astral.sh/github/python-build-standalone/releases/download"),
    ("GitHub", "https://github.com/astral-sh/python-build-standalone/releases/download"),
    ("NJU", "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"),
    ("USTC", "https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone"),
)
PYPI_MIRRORS = (
    ("PyPI", "https://files.pythonhosted.org/packages"),
    ("Tsinghua", "https://pypi.tuna.tsinghua.edu.cn/packages"),
    ("Aliyun", "https://mirrors.aliyun.com/pypi/packages"),
    ("Huawei", "https://repo.huaweicloud.com/repository/pypi/packages"),
)
TORCH_MIRRORS = (
    ("PyTorch R2", "https://download-r2.pytorch.org/whl"),
    ("PyTorch", "https://download.pytorch.org/whl"),
    ("Aliyun", "https://mirrors.aliyun.com/pytorch-wheels"),
    ("NJU", "https://mirror.nju.edu.cn/pytorch/whl"),
)
MirrorList = Iterable[str | tuple[str, str]]


def safe_url(value: str) -> str:
    """日志只保留下载地址的主机和路径，不输出认证与查询参数。"""
    try:
        parsed = urllib.parse.urlsplit(value)
        authority = parsed.netloc.rsplit("@", 1)[-1]
        return urllib.parse.urlunsplit((parsed.scheme, authority, parsed.path, "", ""))
    except ValueError:
        return "<invalid URL>"


def normalize_root(value: str) -> str:
    """校验下载根，保留 HTTP 以支持内网镜像和本地行为验证。"""
    if not isinstance(value, str) or not value.strip() or any(character.isspace() for character in value):
        raise ValueError("A download root must be a non-empty HTTP or HTTPS URL without whitespace.")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError(f"Invalid download root: {safe_url(value)}")
    if parsed.port is not None and not 0 < parsed.port < 65536:
        raise ValueError(f"Invalid download root port: {safe_url(value)}")
    return value.rstrip("/")


@dataclass(frozen=True)
class DownloadSource:
    name: str
    root: str
    urls: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", normalize_root(self.root))
        object.__setattr__(self, "urls", tuple(self.urls))
        if not self.urls:
            raise ValueError("A download source must contain at least one artifact URL.")
        for url in self.urls:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"Invalid artifact URL: {safe_url(url)}")


@dataclass(frozen=True)
class ProbeResult:
    source: DownloadSource
    elapsed: float
    bytes_read: int
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and self.bytes_read > 0

    @property
    def speed(self) -> float:
        return self.bytes_read / self.elapsed if self.available and self.elapsed > 0 else 0.0


def _safe_error(exc: Exception) -> str:
    message = re.sub(r"https?://[^\s'\"<>]+", lambda match: safe_url(match.group()), str(exc))
    return f"{type(exc).__name__}: {message}"


# ---------- 前缀测速 ----------

def _artifact_magic(url: str) -> bytes:
    filename = urllib.parse.unquote(urllib.parse.urlsplit(url).path).lower()
    for suffixes, magic in (
        ((".whl", ".zip"), b"PK\x03\x04"),
        ((".gz", ".tgz"), b"\x1f\x8b"),
        ((".xz",), b"\xfd7zXZ\x00"),
        ((".zst", ".zstd"), b"\x28\xb5\x2f\xfd"),
    ):
        if filename.endswith(suffixes):
            return magic
    raise ValueError("The artifact URL must identify a wheel or a supported compressed archive.")


def _read_prefix(url: str, timeout: float, probe_bytes: int) -> int:
    magic = _artifact_magic(url)
    if probe_bytes < len(magic):
        raise ValueError("The probe size is too small to validate the artifact header.")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{probe_bytes - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "quadruped-lab/setup-downloads",
        },
    )
    deadline = time.monotonic() + timeout
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status not in {200, 206}:
            raise ValueError(f"Unexpected HTTP status: {response.status}.")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise ValueError("The server changed the requested artifact encoding.")
        content_length = response.headers.get("Content-Length")
        expected_size = min(int(content_length), probe_bytes) if content_length is not None else None
        if expected_size is not None and expected_size <= 0:
            raise ValueError("The artifact response is empty.")
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            matched = re.fullmatch(r"bytes 0-(\d+)/(\d+|\*)", content_range.strip())
            if matched is None:
                raise ValueError("The server returned an invalid prefix Content-Range.")
            range_end = int(matched[1])
            if range_end >= probe_bytes or (matched[2] != "*" and int(matched[2]) <= range_end):
                raise ValueError("The server returned a range outside the requested prefix.")
            if expected_size is not None and expected_size != range_end + 1:
                raise ValueError("Content-Length and Content-Range disagree.")
            expected_size = range_end + 1

        payload = bytearray()
        while len(payload) < probe_bytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Artifact prefix download exceeded its time limit.")
            # HTTPResponse 的 read1 每次至多触发一次底层读取，避免慢速流不断重置总时限。
            stream = getattr(response, "fp", None)
            raw = getattr(stream, "raw", None)
            socket = getattr(raw, "_sock", None)
            if socket is not None:
                socket.settimeout(remaining)
            chunk = response.read1(min(65536, probe_bytes - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) >= len(magic) and not payload.startswith(magic):
                raise ValueError("The response is not the requested wheel or archive (possibly an HTML page).")
            if expected_size is not None and len(payload) >= expected_size:
                break
        if time.monotonic() > deadline:
            raise TimeoutError("Artifact prefix download exceeded its time limit.")
        if not payload.startswith(magic):
            raise ValueError("The artifact prefix is empty or has an invalid header.")
        if expected_size is not None and len(payload) != expected_size:
            raise ValueError("The artifact response ended before its declared prefix length.")
        return len(payload)


def probe_sources(
    sources: Iterable[DownloadSource], *, timeout: float = 8.0, probe_bytes: int = 524288, workers: int = 4
) -> list[ProbeResult]:
    """并发测速；同一源的多个文件全部通过后才按总字节与总耗时排序。"""
    if not math.isfinite(timeout) or timeout <= 0 or probe_bytes <= 0 or workers <= 0:
        raise ValueError("Probe timeout, byte limit, and worker count must be positive.")
    candidates = list(sources)
    if not candidates:
        return []

    def probe(source: DownloadSource) -> ProbeResult:
        started = time.monotonic()
        bytes_read = 0
        try:
            for url in source.urls:
                bytes_read += _read_prefix(url, timeout, probe_bytes)
            return ProbeResult(source, time.monotonic() - started, bytes_read)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            return ProbeResult(source, time.monotonic() - started, bytes_read, _safe_error(exc))

    with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
        results = list(executor.map(probe, candidates))
    return sorted(results, key=lambda result: (not result.available, -result.speed))


def print_probes(title: str, results: Iterable[ProbeResult]) -> None:
    """输出测速结果，下载地址中的认证和查询信息始终隐藏。"""
    print(title, flush=True)
    for position, result in enumerate(results, 1):
        status = (
            f"{result.speed / 1048576:.2f} MiB/s, {result.elapsed:.2f}s, {result.bytes_read} bytes"
            if result.available
            else f"unavailable ({result.error})"
        )
        print(f"  {position}. {result.source.name}: {safe_url(result.source.root)} - {status}", flush=True)


# ---------- Python 准备 ----------

def _run_uv(command: list[str], timeout: float, *, capture: bool = True) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        if capture:
            stdout, stderr = process.communicate(timeout=timeout)
        else:
            process.wait(timeout=timeout)
            stdout, stderr = None, None
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # 超时或用户中断时同时终止 uv 的辅助进程，继续处理异常前等待清理完成。
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+(?:\.\d+){1,2}", version):
        raise ValueError("Python version must contain two or three numeric components.")


def _find_python(uv: str, version: str, timeout: float) -> str | None:
    command = [uv, "python", "find", version, "--managed-python", "--no-python-downloads", "--no-project", "--system"]
    result = _run_uv(command, timeout)
    path = (result.stdout or "").strip()
    return path if result.returncode == 0 and path and Path(path).is_file() else None


def python_download_sources(uv: str, version: str, mirrors: MirrorList | None = None) -> list[DownloadSource]:
    """从 uv 获取匹配版本的真实发行包，再生成各镜像的具体文件地址。"""
    _validate_version(version)
    result = _run_uv(
        [uv, "python", "list", version, "--only-downloads", "--show-urls", "--output-format", "json"], 8.0
    )
    if result.returncode != 0:
        raise RuntimeError("uv could not list the requested Python download.")
    try:
        downloads = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise RuntimeError("uv returned invalid Python download metadata.") from exc
    asset_url = None
    for download in downloads:
        if (
            download.get("implementation") == "cpython"
            and download.get("variant", "default") == "default"
            and not download.get("prerelease")
            and str(download.get("version", "")).split(".")[:len(version.split("."))] == version.split(".")
            and download.get("url")
        ):
            asset_url = str(download["url"])
            break
    if asset_url is None:
        raise RuntimeError(f"uv did not report a stable CPython {version} download.")
    suffix = urllib.parse.urlsplit(asset_url).path.partition("/releases/download/")[2]
    if not suffix:
        raise RuntimeError("uv returned an unsupported Python download URL layout.")
    if mirrors is None:
        configured = os.environ.get("UV_PYTHON_INSTALL_MIRROR")
        mirrors = [("Configured Python mirror", configured)] if configured else PYTHON_MIRRORS
    sources = []
    for index, mirror in enumerate(mirrors, 1):
        name, root = (f"Custom Python mirror {index}", mirror) if isinstance(mirror, str) else mirror
        root = normalize_root(root)
        sources.append(DownloadSource(name, root, (f"{root}/{suffix}",)))
    if not sources:
        raise ValueError("At least one Python download source is required.")
    return sources


def ensure_python(
    uv: str,
    version: str,
    *,
    mirrors: MirrorList | None = None,
    offline: bool = False,
    timeout: float = 8.0,
    probe_bytes: int = 524288,
    install_timeout: float = 600.0,
) -> str:
    """复用 uv Python，或按测速结果依次安装；离线时不触发网络请求。"""
    _validate_version(version)
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(install_timeout) or install_timeout <= 0:
        raise ValueError("Python lookup and installation timeouts must be positive and finite.")
    interpreter = _find_python(uv, version, timeout)
    if interpreter is not None:
        print(f"Reusing managed Python {version}: {interpreter}", flush=True)
        return interpreter
    if offline or os.environ.get("UV_OFFLINE", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"Managed Python {version} is not installed; offline mode cannot download it.")
    results = probe_sources(python_download_sources(uv, version, mirrors), timeout=timeout, probe_bytes=probe_bytes)
    print_probes(f"Python {version} download source probes:", results)
    available = [result for result in results if result.available]
    if not available:
        raise RuntimeError("No Python download source passed the artifact probe.")
    for position, result in enumerate(available, 1):
        print(f"Installing Python {version} ({position}/{len(available)}) from {safe_url(result.source.root)}", flush=True)
        try:
            installed = _run_uv(
                [uv, "python", "install", version, "--mirror", result.source.root], install_timeout, capture=False
            )
        except subprocess.TimeoutExpired:
            print("Python installation timed out; trying the next available source.", flush=True)
            continue
        if installed.returncode == 0:
            interpreter = _find_python(uv, version, timeout)
            if interpreter is not None:
                return interpreter
            print("Python installation did not produce a discoverable interpreter; trying the next source.", flush=True)
        else:
            print(f"Python installation failed with exit code {installed.returncode}; trying the next source.", flush=True)
    raise RuntimeError(f"Python {version} installation failed for all available download sources.")
