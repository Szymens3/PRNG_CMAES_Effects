#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python run_experiments.py --file_depended
python run_experiments.py 
