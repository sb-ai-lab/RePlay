#!/bin/bash

# TODO: запросить у техподдержки Jenkins обновить Python (сейчас 3.6.8)
python3 -m venv venv
source ./venv/bin/activate
# TODO: pypandoc нужен для pyspark, cython для implicit но почему-то некорректно взаимодействуют с poetry
pip install --index-url http://mirror.sigma.sbrf.ru/pypi/simple \
    --trusted-host mirror.sigma.sbrf.ru -U poetry pip pypandoc cython optuna
# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install
