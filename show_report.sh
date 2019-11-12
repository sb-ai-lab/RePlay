#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
pycodestyle ${PACKAGE_NAME} tests
mypy --ignore-missing-imports ${PACKAGE_NAME} tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
coverage run --source ${PACKAGE_NAME} -m unittest discover -s tests/
coverage report -m
cloc ${PACKAGE_NAME}
