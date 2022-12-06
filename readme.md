# About
Group project of COMP4471@HKUST in 2022 Fall.

# Environment
`conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=11.7 -c pytorch -c nvidia`


# Training
In the root directory of project.
Notice that, the random seed is based on the configuration file content, to ensure reproducability and also prevent overfit.

## Single processing mode:
Turn off the distributed.toggle in conf.yaml
```bash
CUDA_VISIBLE_DEVICES=3 python main.py conf_file=conf.yaml
```

To run in test mode:
```bash
CUDA_VISIBLE_DEVICES=3 python main.py schedule.test_only=True conf_file=./exps/static/noWdecay_LR7_OneCycle_S/conf_backup.yaml
```

## Distributed mode
FYI: batch size of 6 needs 10GB GPU memory
(single node, replace `4` into the number of GPU you want to use):

1. torchrun
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py conf_file=conf.yaml distributed.toggle=True
    ```
    For the test mode: `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py distributed.toggle=True schedule.test_only=True conf_file=conf.yaml `
2. or torch multiprocessing spawn (not recommended because of termination messy)
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,3 python main.py conf_file=conf.yaml distributed.toggle=True distributed.gpu_workers=3
    ```

# Tensorboard
To **visualize**(in the root directory of project):
`pip install tensorboard`
`tensorboard --logdir=exps`
then open the given link from terminal in the browser

# Implementation Reference
https://github.com/selimsef/dfdc_deepfake_challenge

Distributed training:
1. https://pytorch.org/docs/stable/elastic/train_script.html
2. https://leimao.github.io/blog/PyTorch-Distributed-Training/
3. Pytorch Elastic: https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py, https://pytorch.org/docs/stable/elastic/run.html
4. (future) https://pytorch.org/docs/stable/distributed.algorithms.join.html, https://discuss.pytorch.org/t/question-about-init-process-group/107912, https://murphypei.github.io/blog/2021/05/torch-barrier-trap