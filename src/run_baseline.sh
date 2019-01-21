#!/bin/bash

python3 baseline.py
python3 prior.py -r 0
python3 cdt.py ./logs/ -r 0 --dcdt --rcdt --baseline