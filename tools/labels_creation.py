from __future__ import print_function

import sys
sys.path.append("config")
sys.path.append("blazeface")

import torch
import argparse
import torch.utils.data as data
from data.dataset.wider_face import WiderFaceDetection, detection_collate
from data.transform.data_augment import preproc
from config import cfg_blaze
from models.module.prior_box import PriorBox
import logging

LOG_FORMAT = "%(asctime)s %(levelname)s : %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--training_dataset', default='./blazeface/data/dataset/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')

args = parser.parse_args()
cfg = cfg_blaze


rgb_mean = (104, 117, 123) # bgr order
img_dim = cfg['image_size']
batch_size = cfg['batch_size']


def labeling():
    num_workers = args.num_workers
    training_dataset = args.training_dataset

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()


    logging.info('Loading Dataset...')
    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    loader = data.DataLoader(dataset, batch_size, 
                            shuffle=True, num_workers=num_workers, 
                            collate_fn=detection_collate, pin_memory=True)

if __name__ == '__main__':
    labeling()
