import torch
from .dataset import DfdcDataset
from .augment import create_transforms_train, create_transforms_val
from torch.utils.data import DataLoader

# https://pytorch.org/docs/stable/data.html
def configure_data(args, config):
    # Transforms (augmentation, converting img to tensor, etc.)
    trans_train = create_transforms_train(config['size'])
    trans_val = create_transforms_val(config['size'])

    # Map-style dataset
    data_train = DfdcDataset(mode='train',
                             root_dir=args.root_dir,
                             crops_dir=args.crops_dir,
                             fold=args.fold,
                             folds_csv_path=args.folds_csv_path,
                             folds_json_path=args.folds_json_path,
                             trans=trans_train)
    data_val = DfdcDataset(mode='val',
                             root_dir=args.root_dir,
                             crops_dir=args.crops_dir,
                             fold=args.fold,
                             folds_csv_path=args.folds_csv_path,
                             folds_json_path=args.folds_json_path,
                             trans=trans_val)

    # Customized sampler
    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
    # https://pytorch.org/docs/stable/distributed.html#distributed-launch
    # https://pytorch.org/tutorials/beginner/dist_overview.html#data-parallel-training
    sampler_train = None
    sampler_val = None
    if args.is_distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(data_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(data_val)

    # Loader
    loader_train = DataLoader(data_train,
                              batch_size=config['optimizer']['batch_size'],
                              shuffle=sampler_train is None,
                              sampler=sampler_train,
                              pin_memory=False,
                              # num_workers=args.workers,
                              # collate_fn=lambda x: x
                              )

    loader_val = DataLoader(data_val,
                            batch_size=config['optimizer']['batch_size'] * 2, # ???
                            sampler=sampler_val,
                            shuffle=False,
                            pin_memory=True,
                            # num_workers=args.workers,
                            # collate_fn=lambda x: x
                            )
    return sampler_train, loader_train, loader_val
