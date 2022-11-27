"""
utils for preprocessing

file structure:
--video_dir
 |--dfdc_train_part_0
   |--metadata.json
 |--dfdc_train_part_1
 |--...

"""
import os
import json
from glob import glob
from pathlib import Path

def get_real_video_paths(video_dir, basename=False):
    real_paths = set()
    real_names = set()
    for json_path in glob(os.path.join(video_dir, "*/metadata.json")):
        dir = Path(json_path).parent # dfdc_train_part_i
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for name, v in metadata.items():
            # {name: value} where name = ".mp4",
            #                     value = {"label": , "split": , "real": }
            real = v.get("real", None)
            if v["label"] == "REAL":
                real_names.add(name)
                real_paths.add(os.path.join(dir, name))
    real_paths = list(real_paths)
    real_names = list(real_names)
    print(f'\# of real videos: {len(real_paths)}')
    return real_names if basename else real_paths

def get_real_with_fakes(video_dir):
    pairs = []
    for json_path in glob(os.path.join(video_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs
