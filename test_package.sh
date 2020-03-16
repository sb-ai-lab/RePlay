#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
source ./venv/bin/activate
pycodestyle --ignore=E501,W605,W504 --max-doc-length=160 ${PACKAGE_NAME} tests
coverage run --source ${PACKAGE_NAME} -m pytest -s tests/
coverage report -m
cd docs
mkdir -p _static
make clean html
