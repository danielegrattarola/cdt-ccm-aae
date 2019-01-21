#!/bin/bash

python3 src/main_prior.py
python3 src/main_geom.py
python3 src/main_cdt.py logs/ --dcdt --rcdt