import cv2  # for capturing videos
import math  # for mathematical operations
import pandas as pd
import numpy as np
import argparse
import os
from glob import glob
import logging
#
import matplotlib.pyplot as plt
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# URLs and filenames
DIR = "data/face_forensics/benchmark/"
DIR_IMAGES = "faceforensics_benchmark_images"
DIR_FACES = "faces"

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Select a number of videos number to process')
    parser.add_argument('-f', '--format', type=str, default='png', help='image format')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('Running on device: {}'.format(device))
    mount_dirs(args)

    # Create face detector
    mtcnn = MTCNN(
        image_size=160, margin=25, min_face_size=45,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, select_largest=True, keep_all=False
    )

    tags = get_video_tags(args)
    save_face(args, mtcnn, tags)


def save_face(args, mtcnn, tags):
    # storing the frames from videos
    for tag in tqdm(tags):
        img = Image.open(tag)
        tag_face = tag.replace(DIR_IMAGES,DIR_FACES)
        mtcnn(img, tag_face)


def get_video_tags(args):
    tags = glob(os.path.join(DIR,DIR_IMAGES,'*.' + args.format))
    return tags


def mount_dirs(args):
    try:
        os.mkdir(os.path.join(DIR,DIR_FACES))
    except FileExistsError:
        pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
