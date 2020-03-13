#!/bin/bash

# TODO: запросить у техподдержки Jenkins обновить Python (сейчас 3.6.8)
python3 -m venv venv
source ./venv/bin/activate
# TODO: nose нужен для annoy, pypandoc для pyspark. Оба почему-то некорректно взаимодействуют с poetry
pip install --index-url http://mirror.sigma.sbrf.ru/pypi/simple \
    --trusted-host mirror.sigma.sbrf.ru -U poetry pip nose pypandoc
# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install
