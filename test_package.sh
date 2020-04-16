#!/bin/bash

set -e
PACKAGE_NAME=sponge_bob_magic
source ./venv/bin/activate
cd docs
mkdir -p _static
make clean html
cd ..
pycodestyle --ignore=E501,W605,W504 --max-doc-length=160 ${PACKAGE_NAME} tests
export PYTEST_RUNNING=Y
pytest --cov=${PACKAGE_NAME} --cov-report=term-missing \
       --doctest-modules ${PACKAGE_NAME} --cov-fail-under=83 tests
