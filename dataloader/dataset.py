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
                 folds_json_path,
                 root_dir="data/",
                 crops_dir='crops',
                 trans=None,
                 small_fit=0):
        super().__init__
        self.root_dir = root_dir
        self.crops_dir = crops_dir
        self.mode = mode
        self.trans = trans
        self.trans_totensor = create_transforms_totensor()

        with open(folds_json_path) as openfile:
            self.folds_json = json.load(openfile)
        self.data = self.folds_json[self.mode]
        np.random.shuffle(self.data)
        if small_fit == 1:
            if self.mode == 'train':
                self.data = self.data[:600]
            elif self.mode == 'val':
                self.data = self.data[:30]
            elif self.mode == 'test':
                self.data = self.data[:30]

    def __getitem__(self, index: int):
        # print(f'__getitem__({index})')
        while True:
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
                #video_tensor = img_tensors
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
