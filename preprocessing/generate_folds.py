import argparse
import json
import os
import random
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

from tqdm import tqdm

from utils import get_real_with_fakes

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_frames(entry, root_dir):
    video_name, values = entry
    label, ori = values
    frames = []
    for frame in range(0, 320, 10):
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            fake_img_path = os.path.join(root_dir, 'crops', video_name, image_id)
            ori_img_path = os.path.join(root_dir, 'crops', ori, image_id)
            img_path = ori_img_path if label == 0 else fake_img_path
            if os.path.exists(img_path):
                frames.append(image_id)
    return [video_name, label, ori, frames]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Folds")
    parser.add_argument("--root-dir", help="video directory", default="data/")
    parser.add_argument("--seed", type=int, default=777, help="Seed to split, default 777")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    video_dict_all = {}
    for d in os.listdir(args.root_dir):
        if not os.path.isdir(os.path.join(args.root_dir, d)):
            continue
        if not "dfdc" in d:
            continue
        part = int(d.split("_")[-1])
        for f in os.listdir(os.path.join(args.root_dir, d)):
            if "metadata.json" not in f:
                continue
            with open(os.path.join(args.root_dir, d, "metadata.json")) as metadata_json:
                metadata = json.load(metadata_json)
            for video_file, values in metadata.items():
                video_id = video_file[:-4] # slice ".mp4"
                label = values['label']
                ori = video_id
                label = 0 if label == 'FAKE' else 1
                if label == 0:
                    ori = values['original'][:-4]
                video_dict_all[video_id] = (label, ori)
    video_list_all = list(video_dict_all.items())
    np.random.shuffle(video_list_all)

    video_list_train = []
    video_list_val = []
    video_list_test = []
    for i, entry in enumerate(video_list_all):
        if i % 10 == 7 or i % 10 == 8:
            video_list_val.append(entry)
        elif i % 10 == 9:
            video_list_test.append(entry)
        else:
            video_list_train.append(entry)
    json_train = []
    json_val = []
    json_test = []
    with Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(video_list_train)) as pbar:
            func = partial(get_frames, root_dir=args.root_dir)
            for entry in p.imap_unordered(func, video_list_train):
                pbar.update()
                json_train.append(entry)
        with tqdm(total=len(video_list_val)) as pbar:
            func = partial(get_frames, root_dir=args.root_dir)
            for entry in p.imap_unordered(func, video_list_val):
                pbar.update()
                json_val.append(entry)
        with tqdm(total=len(video_list_test)) as pbar:
            func = partial(get_frames, root_dir=args.root_dir)
            for entry in p.imap_unordered(func, video_list_test):
                pbar.update()
                json_test.append(entry)

    json_all = {'train': json_train, 'val': json_val, 'test': json_test}
    with open("folds.json", "w") as outfile:
        json.dump(json_all, outfile)

if __name__ == '__main__':
    main()
