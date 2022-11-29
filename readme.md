# About
Group project of COMP4471@HKUST in 2022 Fall.

# Implementation Reference
https://github.com/selimsef/dfdc_deepfake_challenge

# Run
In the root directory of project:
```bash
python main.py --config conf.json
```

Distributed mode:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port 6666 main.py
--config conf.json --is-distributed
```

# Tensorboard
To visualize(in the root directory of project):
`pip install tensorboard`
`tensorboard --logdir=runs`