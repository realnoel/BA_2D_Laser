#!/bin/bash

module load stack/2024-06 python_cuda/3.11.6

# Create venv once
python -m venv ~/venvs/ddpm_venv

# Activate
source ~/venvs/ddpm_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
