#!/bin/bash

# create virtual environment
python3 -m venv .env

# activate virtual environment
source .env/bin/activate

#install externals
pip install -r requirements.txt
