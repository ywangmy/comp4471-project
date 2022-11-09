#!/bin/bash
#SBATCH -p cpu3
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -N 1
python slave_crop.py "http://192.168.86.54:10087" 2
