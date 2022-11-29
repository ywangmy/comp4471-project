import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from .augment import create_transforms_totensor

class DfdcDataset(Dataset):

    def __init__(self,
                 mode,
                 root_dir="data/",
                 crops_dir='crops',
                 fold=0,
                 folds_csv_path="folds.csv",
                 trans=None):
        super().__init__
        self.root_dir = root_dir
        self.crops_dir = crops_dir
        self.fold = fold
        self.folds_csv_path = folds_csv_path
        self.folds_csv = pd.read_csv(self.folds_csv_path)
        self.mode = mode
        self.trans = trans
        self.trans_totensor = create_transforms_totensor()
        self.next_epoch()

    def next_epoch(self):
        # k fold
        folds_csv = self.folds_csv
        if self.mode == 'train':
            rows = folds_csv[folds_csv["fold"] != self.fold]
        elif self.mode == 'val':
            rows = folds_csv[folds_csv["fold"] == self.fold]

        self.data = rows.value
        np.random.shuffle(self.data)
        self.epoch += 1

    def __getitem__(self, index: int):
        while True:
            video, img_file, label, ori_video, frame, fold = self.data[index]
            try:
                img_path = os.path.join(self.root_dir, self.crops_dir, video, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Nonw
                if self.trans is not None:
                    results = self.trans(image=image)
                    image, mask = results['image'], results['mask']
                img_tensor = trans_totensor(image=image)['image']
                return {"image": image,
                        "labels": np.array((label,)),
                        "img_name": os.path.join(video, img_file),
                        "valid": valid_label,
                        "rotations": rotation}
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Broken image", os.path.join(self.data_root, self.crops_dir, video, img_file))
                index = random.randint(0, len(self.data) - 1)

    def __len__(self):
        return len(self.data)
