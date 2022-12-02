# About
Group project of COMP4471@HKUST in 2022 Fall.

# Run
In the root directory of project:
```bash
python main.py --comment comment_for_tensorboard
```

Distributed mode(single node, replace 4 into the number of GPU you have):
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py
--comment comment_for_tensorboard --is-distributed
```

# Tensorboard
To visualize(in the root directory of project):
`pip install tensorboard`
`tensorboard --logdir=runs`

# Implementation Reference
https://github.com/selimsef/dfdc_deepfake_challenge

Distributed training:
1. https://pytorch.org/docs/stable/elastic/train_script.html
2. https://leimao.github.io/blog/PyTorch-Distributed-Training/
3. Pytorch Elastic: https://github.com/pytorch/elastic
