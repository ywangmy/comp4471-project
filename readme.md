# About
Group project of COMP4471@HKUST in 2022 Fall.

The final report is in the file: [**COMP4471Report.pdf**](./COMP4471Report.pdf)

# Environment
With CUDA 11.7:
```bash
conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=11.7 -c pytorch -c nvidia
conda env create -f requirement.yml
```
If you encounter other problem, you can refer to [this blog](https://zory233.notion.site/Libraries-Environment-2b2fb672554444b28ebde93c3bde6fc0).

# Training
In the root directory of project.

Notice that, the random seed is based on the core content of configuration file, to ensure reproducability and also prevent overfit.

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
FYI: batch size of 6 needs more than 10GB GPU memory.
The distributed is running in single node, and remember to replace `4` into the number of GPU you want to use.

There're 2 ways to luanch training or test, and the first one prefered to avoid termination messy.

1. **torchrun**
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py distributed.toggle=True conf_file=conf.yaml
    ```
    For the test mode:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py distributed.toggle=True schedule.test_only=True conf_file=conf.yaml
    ```
2. **torch multiprocessing spawn**
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,3 python main.py distributed.toggle=True distributed.gpu_workers=3 conf_file=conf.yaml
    ```
    For the test mode:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,3 python main.py distributed.toggle=True distributed.gpu_workers=3 schedule.test_only=True conf_file=conf.yaml
    ```

# Tensorboard
To **visualize**(in the root directory of project):
```bash
pip install tensorboard
tensorboard --logdir=exps
```
and then open the given link from terminal in the browser.

# Implementation Reference
https://github.com/selimsef/dfdc_deepfake_challenge

Distributed training:
1. https://pytorch.org/docs/stable/elastic/train_script.html
2. https://leimao.github.io/blog/PyTorch-Distributed-Training/
3. Pytorch Elastic: https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py, https://pytorch.org/docs/stable/elastic/run.html
4. (future) https://pytorch.org/docs/stable/distributed.algorithms.join.html, https://discuss.pytorch.org/t/question-about-init-process-group/107912, https://murphypei.github.io/blog/2021/05/torch-barrier-trap

# Note to HKUST students
If you want to use UGCPU cluster, you may refer to [this blog](https://zory233.notion.site/Computing-Resources-Configuration-0e9f1fa889e64567a7e0223405256eb6) and [server.md](./server.md)