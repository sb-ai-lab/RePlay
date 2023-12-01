#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

PROJECT=default
MODE=run
POETRY_ARGS=()

# Parse parameter -p and pass through other paramters
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
      POETRY_ARGS+=("$1")
      shift
      ;;
  esac
done

function cp_if_exists() {
  local src=$1
  local dst=$2

  [ -f $src ] && cp $src $dst
}

# Check if project is valid
case $PROJECT in
  default|experimental) ;;
  *)
    echo "Unknown project '${PROJECT}'"
    exit 1
    ;;
esac

# Patch pyproject.toml
python ${SCRIPT_DIR}/projects/microtemplate.py \
  ${SCRIPT_DIR}/projects/pyproject.toml.template \
  -p project=${PROJECT} \
  > ${SCRIPT_DIR}/pyproject.toml \
  || exit 1

# Copy lock file to the root
cp ${SCRIPT_DIR}/projects/${PROJECT}/poetry.lock ${SCRIPT_DIR}

if [ "${MODE}" = "run" ]; then
  poetry ${POETRY_ARGS[@]}

  # Copy back updated lock file
  cp -u ${SCRIPT_DIR}/poetry.lock ${SCRIPT_DIR}/projects/${PROJECT}

  # Remove generated poetry files
  rm -f ${SCRIPT_DIR}/pyproject.toml
  rm -f ${SCRIPT_DIR}/poetry.lock

  # Sometimes poetry does not cleanup temporary files, so delete this explicitly
  rm -f ${SCRIPT_DIR}/build.py
fi
