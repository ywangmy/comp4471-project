import torch
from torch.utils.data import DataLoader
from .dataset import DfdcDataset
from .augment import create_transforms_train, create_transforms_val

# https://pytorch.org/docs/stable/data.html
def configure_data(cfg):
    # Transforms (augmentation, converting img to tensor, etc.)
    trans_train = create_transforms_train(cfg['dataset']['size'])
    trans_val = create_transforms_val(cfg['dataset']['size'])

    # Map-style dataset
    data_train = DfdcDataset(mode='train',
                            root_dir=cfg['dataset']['root_dir'],
                            crops_dir=cfg['dataset']['crops_dir'],
                            fold=cfg['dataset']['fold'],
                            folds_csv_path=cfg['dataset']['folds_csv_path'],
                            folds_json_path=cfg['dataset']['folds_json_path'],
                            trans=trans_train,
                            small_fit=cfg['dataset']['small_fit'])
    data_val = DfdcDataset(mode='val',
                        root_dir=cfg['dataset']['root_dir'],
                        crops_dir=cfg['dataset']['crops_dir'],
                        fold=cfg['dataset']['fold'],
                        folds_csv_path=cfg['dataset']['folds_csv_path'],
                        folds_json_path=cfg['dataset']['folds_json_path'],
                        trans=trans_val,
                        small_fit=cfg['dataset']['small_fit'])

    # Sampler
    if cfg['distributed']['toggle']:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset=data_train, shuffle=True)
    else:
        sampler_train = None

    # Loader
    loader_train = DataLoader(data_train,
                              batch_size=cfg['optimizer']['batch_size'],
                              shuffle=sampler_train is None,
                              sampler=sampler_train,
                              pin_memory=False,
                              num_workers=cfg['dataset']['load_workers'],
                              )

    loader_val = DataLoader(data_val,
                            batch_size=cfg['optimizer']['batch_size'] * 2, # less memory consumption
                            shuffle=False,
                            pin_memory=True,
                            num_workers=cfg['dataset']['load_workers'],
                            )
    return loader_train, loader_val
