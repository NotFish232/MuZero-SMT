#!/usr/bin/env bash

python3 -m isort --profile black .
python3 -m black .
python3 -m  autoflake --in-place --remove-all-unused-imports --recursive --exclude venv .