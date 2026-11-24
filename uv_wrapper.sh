#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

PROJECT=default
MODE=run
UV_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --experimental)
      PROJECT=experimental
      shift
      ;;
    --generate)
      MODE=generate
      shift
      ;;
    *)
      UV_ARGS+=("$1")
      shift
      ;;
  esac
done

case $PROJECT in
  default|experimental) ;;
  *)
    echo "Unknown project '${PROJECT}'"
    exit 1
    ;;
esac

# Patch pyproject.toml
python3 "${SCRIPT_DIR}/projects/microtemplate.py" \
  "${SCRIPT_DIR}/projects/pyproject.toml.template" \
  -p project=${PROJECT} \
  > "${SCRIPT_DIR}/pyproject.toml"

# Stamp build suffix (CI dev builds), e.g. PACKAGE_SUFFIX=.dev12345
if [ -n "${PACKAGE_SUFFIX:-}" ]; then
  sed -i "0,/^version = /s/^version = \"\([^\"]*\)\"/version = \"\1${PACKAGE_SUFFIX}\"/" "${SCRIPT_DIR}/pyproject.toml"
fi

# Copy lock file to the root
cp "${SCRIPT_DIR}/projects/${PROJECT}/uv.lock" "${SCRIPT_DIR}"

cleanup() {
  rm -f "${SCRIPT_DIR}/pyproject.toml" "${SCRIPT_DIR}/uv.lock"
}

if [ "${MODE}" = "run" ]; then
  trap cleanup EXIT
  uv "${UV_ARGS[@]}"

  # Copy back updated lock file
  cp -u "${SCRIPT_DIR}/uv.lock" "${SCRIPT_DIR}/projects/${PROJECT}/uv.lock"
fi
