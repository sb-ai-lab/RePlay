#!/bin/bash

set -e
PACKAGE_NAME=sponge_bob_magic
source ./venv/bin/activate
cd docs
mkdir -p _static
make clean html
cd ..
pycodestyle --ignore=E501,W605,W504 --max-doc-length=160 ${PACKAGE_NAME} tests
pytest --cov=${PACKAGE_NAME} --cov-report= tests
pytest --cov=${PACKAGE_NAME} --cov-report=term-missing --cov-append \
       --doctest-modules ${PACKAGE_NAME}
