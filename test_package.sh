#!/bin/bash

source ./venv/bin/activate
pycodestyle --ignore=E501,W605,W504 --max-doc-length=160 sponge_bob_magic tests
