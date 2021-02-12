#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate
# Прописываем откуда брать пакеты и устанавливаем то, что через поэтри не ставится
python3 resolve_mirror.py

# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install
