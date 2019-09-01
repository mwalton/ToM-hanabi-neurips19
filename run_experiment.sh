#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./"
python -c "import sys; print(sys.path)"
cmake .
make
pip install absl-py
pip install gin-config
pip install cffi
python agents/rainbow/train.py --base_dir '/floyd/home/' --tf_device '/cpu:*' --belief_level 1 --beta 25.0
