#!/bin/bash

export OPENBLAS_NUM_THREADS=1
set -e
PACKAGE_NAME=sponge_bob_magic
source ./venv/bin/activate
cd docs
mkdir -p _static
make clean html
cd ..
pycodestyle --ignore=E203,E501,W503,W605,E231 --max-doc-length=160 ${PACKAGE_NAME} tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
mypy --ignore-missing-imports ${PACKAGE_NAME} tests
export PYTEST_RUNNING=Y
pytest --cov=${PACKAGE_NAME} --cov-report=term-missing \
       --doctest-modules ${PACKAGE_NAME} --cov-fail-under=94 tests
