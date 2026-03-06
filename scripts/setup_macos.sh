#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_FSL_INSTALL_DIR="${FSLDIR:-${HOME}/fsl}"
DEFAULT_VENV_DIR="${REPO_ROOT}/.venv"
FSL_INSTALL_DIR="${DEFAULT_FSL_INSTALL_DIR}"
VENV_DIR="${DEFAULT_VENV_DIR}"
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_FSL=0
FORCE_FSL_INSTALL=0
DRY_RUN=0
declare -a FSL_EXTRAS=()

log() {
  printf '[setup] %s\n' "$*"
}

die() {
  printf '[setup] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Sets up PigBET on macOS by:
  1. Creating a Python 3.11 virtual environment
  2. Installing Python dependencies from requirements.txt
  3. Installing FSL via the official getfsl.sh installer
  4. Verifying FSL is usable from the shell

Options:
  --python PATH         Python 3.11 executable to use for the venv
  --venv-dir PATH       Virtual environment directory (default: ${DEFAULT_VENV_DIR})
  --fsl-dir PATH        FSL install directory (default: ${DEFAULT_FSL_INSTALL_DIR})
  --fsl-extra NAME      Install an optional FSL component (repeatable)
  --skip-fsl            Skip FSL installation/configuration
  --force-fsl-install   Re-run the FSL installer even if FSL already exists
  --dry-run             Print the actions without executing them
  -h, --help            Show this help text

Examples:
  ./scripts/setup_macos.sh
  ./scripts/setup_macos.sh --python /opt/homebrew/bin/python3.11
  ./scripts/setup_macos.sh --fsl-extra truenet
  ./scripts/setup_macos.sh --skip-fsl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      [[ $# -ge 2 ]] || die "--venv-dir requires a value"
      VENV_DIR="$2"
      shift 2
      ;;
    --fsl-dir)
      [[ $# -ge 2 ]] || die "--fsl-dir requires a value"
      FSL_INSTALL_DIR="$2"
      shift 2
      ;;
    --fsl-extra)
      [[ $# -ge 2 ]] || die "--fsl-extra requires a value"
      FSL_EXTRAS+=("$2")
      shift 2
      ;;
    --skip-fsl)
      SKIP_FSL=1
      shift
      ;;
    --force-fsl-install)
      FORCE_FSL_INSTALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '+'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

ensure_macos() {
  [[ "$(uname -s)" == "Darwin" ]] || die "This setup script is for macOS only."
}

resolve_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
    echo "${PYTHON_BIN}"
    return
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    local py_ver
    py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${py_ver}" == "3.11" ]]; then
      echo "python3"
      return
    fi
  fi

  die "Python 3.11 is required. Install it first, e.g. 'brew install python@3.11', then rerun with --python if needed."
}

verify_python_version() {
  local python_exec="$1"
  local py_ver
  py_ver="$("${python_exec}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "${py_ver}" == "3.11" ]] || die "Expected Python 3.11, but ${python_exec} is ${py_ver}."
}

create_venv() {
  local python_exec="$1"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "Creating virtual environment at ${VENV_DIR}"
    run_cmd "${python_exec}" -m venv "${VENV_DIR}"
  else
    log "Using existing virtual environment at ${VENV_DIR}"
  fi
}

install_python_deps() {
  local venv_python="${VENV_DIR}/bin/python"
  local venv_pip="${VENV_DIR}/bin/pip"

  verify_python_version "${venv_python}"

  log "Upgrading pip/setuptools/wheel"
  run_cmd "${venv_pip}" install --upgrade pip setuptools wheel

  log "Installing Python requirements"
  run_cmd "${venv_pip}" install -r "${REPO_ROOT}/requirements.txt"
}

fsl_is_installed() {
  [[ -f "${FSL_INSTALL_DIR}/etc/fslconf/fsl.sh" ]] && [[ -x "${FSL_INSTALL_DIR}/share/fsl/bin/fslmaths" ]]
}

install_fsl() {
  local installer_url="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh"
  local tmp_dir installer_path
  local -a installer_args

  if fsl_is_installed && [[ "${FORCE_FSL_INSTALL}" -eq 0 ]]; then
    log "FSL already appears to be installed at ${FSL_INSTALL_DIR}; skipping installer"
    return
  fi

  tmp_dir="$(mktemp -d)"
  installer_path="${tmp_dir}/getfsl.sh"
  trap 'rm -rf "${tmp_dir}"' RETURN

  log "Downloading the official FSL installer from ${installer_url}"
  run_cmd curl -fsSL -o "${installer_path}" "${installer_url}"

  log "Running the official FSL installer"
  installer_args=("${FSL_INSTALL_DIR}")
  for extra in "${FSL_EXTRAS[@]}"; do
    installer_args+=(--extra "${extra}")
  done

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '+ %q %q' sh "${installer_path}"
    for arg in "${installer_args[@]}"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  else
    sh "${installer_path}" "${installer_args[@]}"
  fi

  trap - RETURN
  rm -rf "${tmp_dir}"
}

shell_config_path() {
  local shell_name
  shell_name="$(basename "${SHELL:-}")"
  case "${shell_name}" in
    zsh) echo "${HOME}/.zprofile" ;;
    bash|dash) echo "${HOME}/.bash_profile" ;;
    sh|ksh) echo "${HOME}/.profile" ;;
    *)
      echo ""
      ;;
  esac
}

ensure_fsl_shell_config() {
  local shell_rc
  shell_rc="$(shell_config_path)"

  if [[ -z "${shell_rc}" ]]; then
    log "Skipping shell profile update for unsupported shell: ${SHELL:-unknown}"
    return
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ ! -f "${shell_rc}" ]]; then
      printf '+ %q %q\n' touch "${shell_rc}"
    fi
  else
    touch "${shell_rc}"
  fi

  if grep -Fq "${FSL_INSTALL_DIR}/etc/fslconf/fsl.sh" "${shell_rc}" 2>/dev/null; then
    log "Shell profile already contains FSL configuration: ${shell_rc}"
    return
  fi

  log "Adding FSL configuration to ${shell_rc}"
  local config_block
  config_block=$(
    cat <<EOF

# >>> PigBET FSL setup >>>
FSLDIR="${FSL_INSTALL_DIR}"
PATH="\${FSLDIR}/share/fsl/bin:\${PATH}"
export FSLDIR PATH
. "\${FSLDIR}/etc/fslconf/fsl.sh"
# <<< PigBET FSL setup <<<
EOF
  )

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%s\n' "${config_block}"
  else
    printf '%s\n' "${config_block}" >> "${shell_rc}"
  fi
}

load_fsl_into_current_shell() {
  export FSLDIR="${FSL_INSTALL_DIR}"
  export PATH="${FSLDIR}/share/fsl/bin:${PATH}"

  if [[ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]]; then
    # shellcheck disable=SC1090
    . "${FSLDIR}/etc/fslconf/fsl.sh"
  fi
}

verify_fsl() {
  [[ -n "${FSLDIR:-}" ]] || die "FSLDIR is not set"

  local fslmaths_out imcp_out
  fslmaths_out="$("${FSLDIR}/share/fsl/bin/fslmaths" 2>&1 || true)"
  imcp_out="$("${FSLDIR}/share/fsl/bin/imcp" 2>&1 || true)"

  [[ "${fslmaths_out}" == *"Usage: fslmaths"* ]] || die "FSL verification failed: fslmaths did not return the expected help text."
  [[ "${imcp_out}" == *"Usage:"* ]] || die "FSL verification failed: imcp did not return the expected help text."

  log "FSL verification passed"
}

main() {
  ensure_macos

  local python_exec
  python_exec="$(resolve_python)"
  verify_python_version "${python_exec}"

  create_venv "${python_exec}"
  install_python_deps

  if [[ "${SKIP_FSL}" -eq 0 ]]; then
    install_fsl
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      ensure_fsl_shell_config
      load_fsl_into_current_shell
      verify_fsl
    fi
  else
    log "Skipping FSL setup as requested"
  fi

  cat <<EOF

Setup complete.

Next steps:
  1. Activate the virtual environment:
       source "${VENV_DIR}/bin/activate"
  2. If FSL was installed or configured, open a new terminal or run:
       export FSLDIR="${FSL_INSTALL_DIR}"
       export PATH="\${FSLDIR}/share/fsl/bin:\${PATH}"
       . "\${FSLDIR}/etc/fslconf/fsl.sh"
  3. Launch the orientation helper:
       python "${REPO_ROOT}/inference/orientation_helper.py"
EOF
}

main "$@"
