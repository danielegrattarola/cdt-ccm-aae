#!/bin/bash

r='-1.0 0.0 1.0'
python3 main_prior.py -r $r
python3 main_geom.py -r $r
python3 main_cdt.py ./logs/ -r $r --dcdt --rcdt