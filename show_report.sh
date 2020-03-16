#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
pycodestyle --ignore=E501,W605,W504 --max-doc-length=160 ${PACKAGE_NAME} tests
mypy --ignore-missing-imports ${PACKAGE_NAME} tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
export SPARK_LOCAL_IP=127.0.0.1
coverage run --source ${PACKAGE_NAME} -m pytest -s tests/
coverage report -m
cloc ${PACKAGE_NAME}
