import os
import numpy as np
import pandas as pd
import cv2
import traceback
import os
import sys
import random
import json

import torch
from torch.utils.data import Dataset
from .augment import create_transforms_totensor

class DfdcDataset(Dataset):

    def __init__(self,
                 mode,
                 root_dir="data/",
                 crops_dir='crops',
                 fold=0,
                 folds_csv_path=None,
                 folds_json_path=None,
                 trans=None):
        super().__init__
        self.root_dir = root_dir
        self.crops_dir = crops_dir
        self.fold = fold
        assert folds_csv_path == None and folds_json_path == None, \
            'folds_csv_path and folds_json_path cannot be both None'
        self.folds_json_path = folds_json_path
        self.folds_csv_path = folds_csv_path
        #if folds_json_path is not None:
        with open(folds_csv_path) as openfile:
            self.folds_json = json.load(openfile)
        #else:
        #    self.folds_csv = pd.read_csv(self.folds_csv_path)
        self.mode = mode
        self.trans = trans
        self.trans_totensor = create_transforms_totensor()
        self.epoch=0
        self.next_epoch()

    def next_epoch(self):
        """
        Called at initialization
        """
        # k fold
        # folds_csv = self.folds_csv
        self.data = self.folds_json[self.mode]
        # if self.mode == 'train':
        #     rows = folds_csv[folds_csv["fold"] != self.fold]
        # elif self.mode == 'val':
        #     rows = folds_csv[folds_csv["fold"] == self.fold]
        # self.data = rows.values
        # np.random.shuffle(self.data)
        self.epoch += 1

    def __getitem__(self, index: int):
        while True:
            # video_name, frame_name, label, ori_video, frame, fold = self.data[index]
            video_name, label, ori, frames = self.data[index]
            path_common = os.path.join(self.root_dir, self.crops_dir, video_name)
            try:
                img_tensors = []
                for frame_id in frames:
                    img_path = os.path.join(path_common, frame_id)
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.trans is not None:
                        results = self.trans(image=image)
                        image = results['image']
                        # mask = results['mask'] # for self.mask is not None
                        img_tensors.append(self.trans_totensor(image=image)['image'])
                video_tensor = torch.stack(img_tensors, dim=0)
                return {"video": video_tensor,
                        "video_name": video_name,
                        "label": np.array((label,)),
                        "ori": ori}
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Broken image", os.path.join(self.root_dir, self.crops_dir, video_name, img_file))
                index = random.randint(0, len(self.data) - 1)

    def __len__(self):
        return len(self.data)
