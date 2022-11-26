"""
Find face regions using real videos.
The regions for corresponding fake videos are at exactly same positions.
"""

import argparse
import json
import os
from os import cpu_count
from typing import Type

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

#from preprocessing import face_detector, VideoDataset
import face_detector
from face_detector import FacenetDetector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_real_video_paths

def collate_fn_identity(x):
    return x

def find_face_regions(videos, video_dir, detector_cls: Type[VideoFaceDetector]):
    detector = face_detector.__dict__[detector_cls]#(device="cuda:0")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = FacenetDetector(device)
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=cpu_count() - 2, batch_size=1, collate_fn=collate_fn_identity)
    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
        for j, frames in enumerate(batches):
            result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))})
        id = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(video_dir, "boxes")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(
        description="Finding face regions"
    )
    parser.add_argument('--video-dir', help="Video directory") # .video_dir
    parser.add_argument("--detector", help="Detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    args = parser.parse_args()

    real_paths = get_real_video_paths(args.video_dir)
    find_face_regions(real_paths[:2], args.video_dir, args.detector)

if __name__ == '__main__':
    main()
