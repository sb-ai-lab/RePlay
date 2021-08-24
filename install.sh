#!/bin/bash

python3 -m venv venv
. ./venv/bin/activate

python3 resolve_mirror.py # specifies package locations for pip for inner web + installs some packages
poetry install
