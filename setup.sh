#!/usr/bin/env bash
# Copyright (c) 2026-2027 zh
# 内容：准备 uv，测速安装下载通道，并同步指定 PyTorch 后端的独立环境。
# 用法：bash setup.sh [--torch-backend auto|cpu|cu130] [--probe-only] [-- uv sync 参数]

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
torch_backend="auto"
sync_args=()
download_args=()
bootstrap_python_mirrors=()
probe_only=0

print_help() {
  cat <<'EOF'
Usage: bash setup.sh [OPTIONS] [-- UV_SYNC_ARGS...]

Prepare uv and synchronize this project's .venv using uv.lock.

Options:
  --torch-backend BACKEND  auto (default), cpu, or cu130.
  --probe-only            Probe Python/PyPI/Torch channels without installing.
  --skip-probes           Use the original locked uv synchronization.
  --python-mirror ROOT    Python standalone mirror root (repeatable).
  --pypi-mirror ROOT      Artifact root including /packages (repeatable).
  --torch-mirror ROOT     Artifact root preceding /cpu or /cu130 (repeatable).
  --probe-timeout SECONDS Per-artifact probe timeout (default: 8).
  --probe-bytes BYTES     Probe prefix limit (default: 524288).
  --download-timeout SEC  Per-sync attempt timeout (default: 1800).
  -h, --help               Show this help without installing anything.
  --                      Forward remaining arguments to uv sync.

Auto selects cu130 when nvcc reports CUDA Toolkit 13.0 or newer;
otherwise it selects cpu. A CUDA backend still requires a compatible driver.
Project, environment, Python, and backend overrides cannot be forwarded.
Probe-only mode requires uv and the pinned managed Python to be available.
Offline, check, dry-run, and already synchronized environments skip probing.

Examples:
  bash setup.sh
  bash setup.sh --torch-backend cpu
  bash setup.sh --probe-only
  bash setup.sh --torch-backend cu130 -- --offline
EOF
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --torch-backend)
      (($# >= 2)) || fail "--torch-backend requires a value."
      torch_backend="$2"
      shift 2
      ;;
    --torch-backend=*)
      torch_backend="${1#*=}"
      shift
      ;;
    --probe-only)
      probe_only=1
      download_args+=("$1")
      shift
      ;;
    --skip-probes)
      download_args+=("$1")
      shift
      ;;
    --python-mirror)
      (($# >= 2)) || fail "$1 requires a value."
      download_args+=("$1" "$2")
      bootstrap_python_mirrors+=("$2")
      shift 2
      ;;
    --python-mirror=*)
      download_args+=("$1")
      bootstrap_python_mirrors+=("${1#*=}")
      shift
      ;;
    --pypi-mirror|--torch-mirror|--probe-timeout|--probe-bytes|--download-timeout)
      (($# >= 2)) || fail "$1 requires a value."
      download_args+=("$1" "$2")
      shift 2
      ;;
    --pypi-mirror=*|--torch-mirror=*|--probe-timeout=*|--probe-bytes=*|--download-timeout=*)
      download_args+=("$1")
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      sync_args=("$@")
      break
      ;;
    *) fail "Unknown option: $1. Use --help or place uv sync arguments after --." ;;
  esac
done

case "${torch_backend}" in
  auto|cpu|cu130) ;;
  *) fail "Unsupported PyTorch backend: ${torch_backend}. Choose auto, cpu, or cu130." ;;
esac

# 保留后端选择与仓库边界，其他下载、缓存和诊断参数交给 uv 处理。
for sync_arg in "${sync_args[@]}"; do
  case "${sync_arg}" in
    --extra|--extra=*|--all-extras|--no-extra|--no-extra=*|--active|--project|--project=*|--directory|--directory=*|--python|--python=*|-p|-p?*|--script|--script=*|--package|--package=*|--all-packages|--config-file|--config-file=*|--no-config|--frozen|--no-locked|--no-frozen)
      fail "Cannot forward ${sync_arg}: setup.sh controls the project, Python, environment, and backend."
      ;;
  esac
done

for required_file in pyproject.toml uv.lock .python-version tools/setup_env.py tools/setup_downloads.py; do
  [[ -f "${project_root}/${required_file}" ]] || fail "Missing project file: ${required_file}."
done
python_version=""
IFS= read -r python_version < "${project_root}/.python-version" || true
python_version="${python_version%%#*}"
python_version="${python_version//[[:space:]]/}"
[[ -n "${python_version}" ]] || fail ".python-version must contain a Python version."

if [[ "${torch_backend}" == "auto" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    torch_backend="cpu"
    printf '%s\n' 'CUDA Toolkit was not found on PATH; selecting the CPU backend.'
  elif ! nvcc_output="$(nvcc --version 2>&1)"; then
    torch_backend="cpu"
    printf '%s\n' 'nvcc could not run; selecting the CPU backend.'
  elif [[ "${nvcc_output}" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
    cuda_major="${BASH_REMATCH[1]}"
    cuda_minor="${BASH_REMATCH[2]}"
    if ((10#${cuda_major} >= 13)); then
      torch_backend="cu130"
      printf 'Detected CUDA Toolkit %s.%s; selecting the cu130 backend.\n' "${cuda_major}" "${cuda_minor}"
    else
      torch_backend="cpu"
      printf 'CUDA Toolkit %s.%s is older than 13.0; selecting the CPU backend.\n' "${cuda_major}" "${cuda_minor}"
    fi
  else
    torch_backend="cpu"
    printf '%s\n' 'The CUDA Toolkit version could not be parsed; selecting the CPU backend.'
  fi
fi

# 让 uv、官方安装器和 Python 标准库使用同一组代理设置。
if [[ -n "${UV_PROXY:-}" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-${UV_PROXY}}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-${UV_PROXY}}}"
  export http_proxy="${http_proxy:-${HTTP_PROXY}}"
  export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
fi
offline=0
if [[ "${UV_OFFLINE:-}" == "1" || "${UV_OFFLINE:-}" == "true" ]]; then
  offline=1
fi
for sync_arg in "${sync_args[@]}"; do
  [[ "${sync_arg}" != "--offline" ]] || offline=1
done

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
else
  ((offline == 0 && probe_only == 0)) || fail "uv must already be installed for offline or probe-only mode."
  command -v curl >/dev/null 2>&1 || fail "curl is required to install uv. Install uv from https://docs.astral.sh/uv/."
  printf '%s\n' 'uv was not found; installing it from the official installer.'
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${HOME}/.local/bin" sh
  uv_bin="${HOME}/.local/bin/uv"
  [[ -x "${uv_bin}" ]] || fail "uv installation did not create ${uv_bin}."
fi

# 固定项目虚拟环境，避免外部激活状态或环境变量改变安装目标。
export UV_PROJECT_ENVIRONMENT="${project_root}/.venv"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-60}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
cd -- "${project_root}"
if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  bootstrap_python="$(command -v python3)"
else
  printf '%s\n' 'System Python 3.10+ was not found; locating the pinned Python with uv.'
  find_args=(--managed-python --no-python-downloads --no-project --system)
  if ! bootstrap_python="$("${uv_bin}" python find "${python_version}" "${find_args[@]}" 2>/dev/null)"; then
    ((offline == 0 && probe_only == 0)) || fail "The pinned managed Python must already exist for offline or probe-only mode."
    printf '%s\n' 'The pinned Python is not cached; preparing it with uv before artifact probing.'
    if ((${#bootstrap_python_mirrors[@]} == 0)); then
      bootstrap_python_mirrors=("")
    fi
    python_installed=0
    for python_mirror in "${bootstrap_python_mirrors[@]}"; do
      install_args=(python install "${python_version}")
      [[ -z "${python_mirror}" ]] || install_args+=(--mirror "${python_mirror}")
      if "${uv_bin}" "${install_args[@]}"; then
        python_installed=1
        break
      fi
    done
    ((python_installed)) || fail "uv could not prepare Python ${python_version}."
    bootstrap_python="$("${uv_bin}" python find "${python_version}" "${find_args[@]}")"
  fi
fi
if ((probe_only)); then
  printf 'Probing download channels for Python %s and the %s backend.\n' "${python_version}" "${torch_backend}"
else
  printf 'Synchronizing Python %s with the %s backend in %s.\n' "${python_version}" "${torch_backend}" "${UV_PROJECT_ENVIRONMENT}"
fi
"${bootstrap_python}" -B "${project_root}/tools/setup_env.py" --uv "${uv_bin}" --torch-backend "${torch_backend}" "${download_args[@]}" -- "${sync_args[@]}"
printf '%s\n' 'Setup command completed.'
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is installed in ~/.local/bin, but it is not on PATH in your current shell.'
  printf '%s\n' 'Run this command in your current shell before using uv:'
  printf '%s\n' 'export PATH="$HOME/.local/bin:$PATH"'
fi
if ((probe_only == 0)); then
  printf '%s\n' 'Check it with: uv run --no-sync python -B tools/check_install.py'
fi
