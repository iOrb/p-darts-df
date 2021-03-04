import copy
import json
import os
import sys
import time
from glob import glob
import numpy as np
import torch
from tqdm import tqdm

import utils
import genotypes
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
#
from matplotlib import pyplot as plt
from PIL import Image

from model import NetworkDF as Network


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--center_crop', action='store_true', default=False, help='use center crop')
parser.add_argument('--arch', type=str, default='PDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--data_dir', type=str, default='data/face_forensics/benchmark/faces', help='data dir')
parser.add_argument('--model', type=str, default='models/model.pth.tar', help='data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

NUM_CLASSES = 5

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')

    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model.load_state_dict(torch.load(args.model)['state_dict'])
    epoch = torch.load(args.model)['epoch']

    model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    mean, std = [0.5978, 0.4296, 0.3926], [0.2297, 0.1851, 0.1784]
    _ , test_transform  = utils._data_transform_ff(args, size=160, mean=mean, std=std)

    model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    path =  os.path.join(args.data_dir, '*.png')
    img_paths = sorted(glob(path))
    model.eval()
    predictions = {}
    for i, img_path in enumerate(tqdm(img_paths)):
        # show_image_trans(img_path, bench_transform)
        img = Image.open(img_path)
        img = test_transform(img)
        img = img.unsqueeze(0)
        output, _ = model(img)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t().data.cpu().numpy()[0]
        pred[pred < 4] = 1
        pred[pred == 4] = 0
        lbl = 'fake' if pred.item() == 1 else 'real'
        img_name = img_path.split('/')[4]
        predictions[img_name] = lbl
    print('reals:', count_reals(predictions.values()) / len(predictions) * 100, '%')
    with open(os.path.join(args.data_dir, '..', f'bentchmark_labeled.json'), 'w') as fp:
        json.dump(predictions, fp)

def count_reals(labels):
    count = 0
    for lbl in labels:
        if 'real' == lbl:
            count += 1
    return count

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
