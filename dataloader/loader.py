import torch
from dataset import DfdcDataset
from augment import create_transforms_train, create_transforms_val

# https://pytorch.org/docs/stable/data.html
def configure_data(args, config):
    # Transforms (augmentation, converting img to tensor, etc.)
    trans_train = create_transforms_train(config['size'])
    trans_val = create_transforms_val(config['size'])

    # Map-style dataset
    data_train = DfdcDataset(mode='train',
                             root_dir=None,
                             crops_dir=None,
                             fold=None,
                             folds_csv_path=None,
                             trans=None)
    data_val = DfdcDataset(mode='val',
                           root_dir=None,
                           crops_dir=None,
                           fold=None,
                           folds_csv_path=None,
                           trans=None)

    # Customized sampler
    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
    # https://pytorch.org/docs/stable/distributed.html#distributed-launch
    sampler_train = torch.utils.data.distributed.DistributedSampler(data_train)
    sampler_val = torch.utils.data.distributed.DistributedSampler(data_val)

    # Loader
    loader_train = DataLoader(data_train,
                              batch_size=config.batch_size,
                              shuffle=train_sampler is None,
                              sampler=sampler_train,
                              pin_memory=False,
                              # num_workers=args.workers,
                              )

    loader_val = DataLoader(data_val,
                            batch_size=config.batch_size * 2, # ???
                            sampler=sampler_val,
                            shuffle=False,
                            pin_memory=True,
                            # num_workers=args.workers,
                            )
    return sampler_train, loader_train, loader_val