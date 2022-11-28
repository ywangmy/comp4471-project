# About
Group project of COMP4471@HKUST in 2022 Fall.

# Implementation Reference
https://github.com/selimsef/dfdc_deepfake_challenge

# Run
python main.py --config conf.json

later"
python -m torch.distributed.launch --nproc_per_node=4 main.py --config conf.json