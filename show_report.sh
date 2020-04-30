#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
mypy --ignore-missing-imports ${PACKAGE_NAME} tests
cloc ${PACKAGE_NAME}
./test_package.sh
