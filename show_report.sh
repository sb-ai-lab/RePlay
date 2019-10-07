#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
pycodestyle ${PACKAGE_NAME} tests
mypy ${PACKAGE_NAME} tests
pylint ${PACKAGE_NAME} tests
coverage run --source ${PACKAGE_NAME} -m unittest discover -s tests/
coverage report -m
cloc ${PACKAGE_NAME}
