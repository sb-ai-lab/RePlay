#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
source ./venv/bin/activate
pycodestyle --ignore=E501,W605,W504 --max-doc-length=160 ${PACKAGE_NAME} tests
pytest --cov=sponge_bob_magic --cov-report= tests
pytest --cov=sponge_bob_magic --cov-report=term-missing --cov-append --doctest-modules sponge_bob_magic
cd docs
mkdir -p _static
make clean html
