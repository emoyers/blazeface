from __future__ import print_function

import sys
sys.path.append("config")
sys.path.append("blazeface")

import torch
import argparse
import torch.utils.data as data
from data.dataset.wider_face import WiderFaceDetection, WiderFaceDetection2 , detection_collate
from config import cfg_blaze1
from data.transform.data_augment import preproc
from models.module.prior_box import PriorBox
import cv2
import logging
import numpy as np

def scale_point_form(boxes, clip=True):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    img_dim = cfg_blaze1['image_size']
    boxes_scaled = torch.cat((boxes[:, :2] - boxes[:, 2:]/2, # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)      # xmax, ymax
    boxes_scaled *= img_dim
    if clip:
        boxes_scaled.clamp_(max=img_dim-1, min=0)
    return boxes_scaled

def add_bounding_box(img, bbx, color=(255,0,0), thickness=2):
    start_point = (int(bbx[0].item()), int(bbx[1].item()))
    end_point = (int(bbx[2].item()), int(bbx[3].item()))
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img


LOG_FORMAT = "%(asctime)s %(levelname)s : %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--training_dataset', default='./blazeface/data/dataset/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')

args = parser.parse_args()
cfg = cfg_blaze1


rgb_mean = (104, 117, 123) # bgr order
img_dim = cfg['image_size']
batch_size = 1


def labeling():
    num_workers = 0
    training_dataset = args.training_dataset

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()

    # scaling bounding boxes to be plotted in the image
    priors_scaled = scale_point_form(priors)

    logging.info('Loading Dataset...')
    dataset = WiderFaceDetection2(training_dataset)
    logging.info('Loading Dataset2...')
    dataset2 = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    loader = data.DataLoader(dataset, batch_size, 
                             shuffle=False, num_workers=num_workers, 
                             collate_fn=detection_collate, pin_memory=True)

    loader2 = data.DataLoader(dataset2, batch_size, 
                              shuffle=False, num_workers=num_workers, 
                              collate_fn=detection_collate, pin_memory=True)

    for i, tuple in enumerate(loader):
        img, labels = tuple
        # print(img.shape, labels)

        annotations_ = labels[0][0]
        np_img = torch.flatten(img, start_dim=0, end_dim=1).numpy()
        np_img = add_bounding_box(np_img, annotations_)
        
        starting_offset = 4
        for j in range(5):
            x_index = starting_offset + j*2
            y_index = starting_offset + j*2 + 1
            center_coordinates = (int(annotations_[x_index].item()), int(annotations_[y_index].item()))
            np_img = cv2.circle(np_img, center_coordinates, 1, (0,0,255), 3)

        cv2.imshow('image',np_img)
        cv2.waitKey(0)

        if i >= 1:
            break
    
    # print("converted -----------------------")

    for i, tuple in enumerate(loader2):
        img2, labels2 = tuple
        # print(img2.shape, labels2)

        # annotations_ = labels2[0][0]
        # width = w_h[0][0]
        # height = w_h[0][1]

        # boxes = annotations_[:4]
        # landm = annotations_[4:-1]

        # boxes[0::2] *= width
        # boxes[1::2] *= height
        # boxes_np = boxes.type(torch.int).numpy()
        # print(boxes_np)

        # landm[0::2] *= width
        # landm[1::2] *= height

        img2 = torch.flatten(img2, start_dim=0, end_dim=1)
        img2 = img2.permute(1, 2, 0)
        np_img = img2.numpy()

        img_anchors = np.zeros(np_img.shape)
        count_color = 0
        print(priors_scaled.shape[0])
        for i in range(priors_scaled.shape[0]):
            prior =  priors_scaled[i]
            if count_color%3 == 0:
                tuple_color = (255,0,0)
            elif count_color%3 == 1:
                tuple_color = (0,255,0)
            else:
                tuple_color = (0,0,255)

            img_anchors = add_bounding_box(img_anchors, prior, color=tuple_color, thickness=1)
            count_color += 1
        # np_img = cv2.resize(np_img, (width, height), interpolation=cv2.INTER_CUBIC)
        # start_point = (boxes_np[0], boxes_np[1])
        # end_point = (boxes_np[2], boxes_np[3])
        # np_img = cv2.rectangle(np_img, start_point, end_point, (255,0,0), 2)
        cv2.imshow('image',img_anchors)
        cv2.waitKey(0)
        # print("-----------------------")

        if i >= 1:
            break
    
    cv2.destroyWindow('image')
    

if __name__ == '__main__':
    labeling()
