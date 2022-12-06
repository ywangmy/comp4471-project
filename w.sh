CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=3 main.py conf_file=sta_new_mix.yaml
