import random
import sys

import cv2  # for capturing videos
import numpy as np
import argparse
import os
from glob import glob
import logging
import pandas as pd
#
import torch
from PIL import Image
from tqdm import tqdm
import json

# URLs and filenames
USAGES = ['train','test','val']
LABLES = ['real', 'fake']

# MANIPULATION_METHODS = ['pristine', 'neuraltextures']
# MANIPULATION_METHODS_PRUNED = ['p', 'nt']

MANIPULATION_METHODS = {
    'ALL': [['pristine', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures'], ['p', 'df', 'f2f', 'fs', 'nt']],
    'df': [['pristine', 'deepfakes'],['p', 'df']],
    'f2f': [['pristine', 'face2face'],['p', 'f2f']],
    'fs': [['pristine', 'faceswap'],['p', 'fs']],
    'nt': [['pristine', 'neuraltextures'],['p', 'nt']],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--data', type=str, default='data/face_forensics', help='data dir')
    parser.add_argument('-mm', '--manipulation_method', type=str, default='ALL', help='kind of manipulation method preprocessed')
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Select a number of videos number to process')
    parser.add_argument('-ntf', '--num_frames_train_fake', type=int, default=None,
                        help='Select a number of frames per video for train (all if None)')
    parser.add_argument('-ntr', '--num_frames_train_real', type=int, default=None,
                        help='Select a number of frames per video for train (all if None)')
    parser.add_argument('-nv', '--num_frames_test_val', type=int, default=None,
                        help='Select a number of frames per video for test and val (all if None)')
    parser.add_argument('-i', '--input_dir', type=str, default='all_videos', help='input videos dir')
    parser.add_argument('-o', '--out_dir', type=str, default='faces', help='out faces dir')
    parser.add_argument('--splits_dir', type=str, default='splits', help='splits dir')
    parser.add_argument('--bounding_boxes', type=str, default='bounding_boxes', help='bounding boxes dir')
    parser.add_argument('-vf', '--video_format', type=str, default='mp4', help='video format')
    parser.add_argument('-if', '--img_format', type=str, default='jpg', help='image format')
    args = parser.parse_args()
    return args


def main(args):
    num_videos = 0

    splits_dict = {}
    for use in USAGES:
        splits_dict[use] = {LABLES[1]: [],LABLES[0]: []}
        json_split_fn = os.path.join(args.data, args.splits_dir, f'{use}.json')
        with open(json_split_fn) as json_file:
            split_list = json.load(json_file)
        splits_dict[use][LABLES[0]] = list(map(lambda e: f'{str(e[0]).zfill(3)}.{args.video_format}', split_list)) + \
                                   list(map(lambda e: f'{str(e[1]).zfill(3)}.{args.video_format}', split_list))
        splits_dict[use][LABLES[1]] = list(map(lambda e: f'{str(e[0]).zfill(3)}_{str(e[1]).zfill(3)}.{args.video_format}', split_list)) + \
                                   list(map(lambda e: f'{str(e[1]).zfill(3)}_{str(e[0]).zfill(3)}.{args.video_format}', split_list))
        num_videos += len(splits_dict[use][LABLES[0]]) + len(splits_dict[use][LABLES[1]]*(len(MANIPULATION_METHODS[args.manipulation_method][0])-1))

    pbar = tqdm(total=num_videos)
    os.mkdir(os.path.join(args.data, args.out_dir))
    for use in USAGES:
        os.mkdir(os.path.join(args.data, args.out_dir, use))
        out_real_face_dir = os.path.join(args.data, args.out_dir, use, LABLES[0])
        os.mkdir(out_real_face_dir)
        out_fake_face_dir = os.path.join(args.data, args.out_dir, use, LABLES[1])
        os.mkdir(out_fake_face_dir)
        for i, mm in enumerate(MANIPULATION_METHODS[args.manipulation_method][0]):
            lbl = LABLES[0] if mm == 'pristine' else LABLES[1]
            if lbl == LABLES[0]:
                o_dir = out_real_face_dir
                max_frames = args.num_frames_train_real if use == USAGES[0] else args.num_frames_test_val
            else:
                o_dir = out_fake_face_dir
                max_frames = args.num_frames_train_fake if use == USAGES[0] else args.num_frames_test_val

            max_frames = 100 if use == USAGES[1] else 1

            fn_ext = MANIPULATION_METHODS[args.manipulation_method][1][i]
            for fn in splits_dict[use][lbl]:
                video_path = os.path.join(args.data, args.input_dir, mm, fn)
                json_bb_path = os.path.join(args.data, args.bounding_boxes, mm, fn.split('.')[-2] + '.json')
                save_faces(video_path = video_path, max_frames = max_frames, json_bb_path = json_bb_path, out_face_dir = o_dir, fn_extension=fn_ext)
                pbar.update(1)
    pbar.close()
    logging.info(f'Done: {args.data}')



def save_faces(video_path, max_frames, json_bb_path, out_face_dir, fn_extension = None):
    # Read and write
    reader = cv2.VideoCapture(video_path)

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fn = video_path.split('/')[-1].split('.')[0]

    batch_size = max_frames if max_frames and max_frames <=num_frames else num_frames

    try:
        with open(json_bb_path) as json_file:
            df = json.load(json_file)
    except:
        print("\nAnalysis", f'{video_fn}.mp4 FAILED')
        sys.exit(1)

    miss_face, frame_num = 0, 0

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break

        frame_num_str = str(frame_num).zfill(4)

        try:
            x = df[frame_num_str][0]
            y = df[frame_num_str][1]
            size = int(df[frame_num_str][2])
        except:
            miss_face += 1
            frame_num += 1
            continue

        cropped_face = image[y:y + size, x:x + size]
        # show_image(cropped_face)

        if max_frames:
            if random.random() >= 0.15:
                out_tag = os.path.join(out_face_dir,f'%s{video_fn}_frame{frame_num}.{args.img_format}' % (fn_extension + '_' if fn_extension else ''))
                cv2.imwrite(out_tag, cropped_face)
                frame_num += 1
        else:
            out_tag = os.path.join(out_face_dir,f'%s{video_fn}_frame{frame_num}.{args.img_format}' % (fn_extension + '_' if fn_extension else ''))
            cv2.imwrite(out_tag, cropped_face)
            frame_num += 1

        if frame_num >= batch_size:
            break

if __name__ == '__main__':
    args = parse_args()
    main(args)
