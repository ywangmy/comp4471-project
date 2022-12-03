# About
Group project of COMP4471@HKUST in 2022 Fall.

# Run
In the root directory of project:

Normal mode:
```bash
python main.py --comment comment_for_tensorboard
```

Distributed mode(single node, replace 4 into the number of GPU you want to use):
```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --is-distributed --comment comment_for_tensorboard
```
or
(not recommended because of termination messy)
```bash
python main.py --is-distributed --gpu-workers 4 --comment comment_for_tensorboard
```
FYI: batch size of 6 needs 10GB GPU memory

For other arguments:
`--workers 3`: number of data loading cpu threads
`--ckpt_path './ckpt/default.pth.tar'`: number of data loading cpu threads

# Tensorboard
To **visualize**(in the root directory of project):
`pip install tensorboard`
`tensorboard --logdir=runs`
open the link given in the browser

# Implementation Reference
https://github.com/selimsef/dfdc_deepfake_challenge

Distributed training:
1. https://pytorch.org/docs/stable/elastic/train_script.html
2. https://leimao.github.io/blog/PyTorch-Distributed-Training/
3. Pytorch Elastic: https://github.com/pytorch/elastic, https://pytorch.org/docs/stable/elastic/run.html
4. (future) https://pytorch.org/docs/stable/distributed.algorithms.join.html, https://discuss.pytorch.org/t/question-about-init-process-group/107912, https://murphypei.github.io/blog/2021/05/torch-barrier-trap
