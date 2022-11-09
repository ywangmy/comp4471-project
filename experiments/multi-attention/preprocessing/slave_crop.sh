#!/bin/bash
#SBATCH -p cpu3
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -N 1
python slave_crop.py "http://10.89.213.69:10087" 2
