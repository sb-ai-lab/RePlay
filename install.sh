#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate
# Прописываем откуда брать пакеты и возвращаем строку установки того, что через
# поэтри не  ставится
install_command = $(python3 resolve_mirror.py)
eval $install_command
# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install
