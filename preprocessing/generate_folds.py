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

def get_frames(video_name, root_dir, crops_dir):
    frames = []
    for frame in range(0, 320, 10):
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            img_path = os.path.join(root_dir, crops_dir, video_name, image_id)
            if os.path.exists(img_path):
                frames.append(image_id)
    if len(frames) > 30:
        frames = frames[:30]
    return frames

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Folds")
    parser.add_argument("--root-dir", help="video directory", default="data/")
    parser.add_argument("--crops-dir", help="crops directory")
    parser.add_argument("--out", help="output file")
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
    """
    video_list_train = []
    video_list_val = []
    video_list_test = []
    """
    video_list_30 = []
    cnt_video = 0
    json_train = []
    json_val = []
    json_test = []
    for video_name, (label, ori) in tqdm(video_list_all):
        frames = get_frames(video_name, args.root_dir, args.crops_dir)
        if len(frames) != 30:
            continue
        cnt_video += 1
        entry = [video_name, label, ori, frames]
        if cnt_video % 10 == 7 or cnt_video % 10 == 8:
            json_val.append(entry)
        elif cnt_video % 10 == 9:
            json_test.append(entry)
        else:
            json_train.append(entry)
    """
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
    """
    json_all = {'train': json_train, 'val': json_val, 'test': json_test}
    with open(args.out, "w") as outfile:
        json.dump(json_all, outfile)
    print(f'train: {len(json_train)}, val: {len(json_val)}, test: {len(json_test)}')

if __name__ == '__main__':
    main()
