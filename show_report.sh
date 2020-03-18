#!/bin/bash

PACKAGE_NAME=sponge_bob_magic
mypy --ignore-missing-imports ${PACKAGE_NAME} tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
export SPARK_LOCAL_IP=127.0.0.1
cloc ${PACKAGE_NAME}
./test_package.sh
