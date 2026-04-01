#!/usr/bin/env bash

# Fix weird checksum errors
./venv/bin/tensorboard --logdir results/ --extra_data_server_flags=--no-checksum