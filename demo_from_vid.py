import sys
import os
current_dir = os.path.dirname(__file__)

# Append the desired directory to sys.path
sys.path.append(os.path.join(current_dir, 'core'))


import argparse
import os
import cv2
import glob
import numpy as np
import torch
import copy
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import time


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def to_tensor(frame):
    img = np.asarray(frame, dtype='int32').astype(np.uint8)
    img = torch.from_numpy(frame).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    # Mag thresholding...
    i = 1
    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2HSV)
    #ret, flo[:,:,i] = cv2.threshold(flo[:,:,i], 80, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', img/255)
    cv2.imshow('flow0', flo[:,:,0])
    cv2.imshow('flow1', flo[:,:,1])
    cv2.imshow('flow2', flo[:,:,2])
    cv2.waitKey(1)


def viz2(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(1)


def vid_demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    #cap = cv2.VideoCapture(args.path)

    cap = cv2.VideoCapture(0)
    freq = 10
    period = 1/freq

    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    ret, old_frame = cap.read()
    with torch.no_grad():
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("End of vid...")
                break
            image1 = to_tensor(old_frame) # Might require a conversion here...
            image2 = to_tensor(frame)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz2(image1, flow_up)
            old_frame = copy.deepcopy(frame) # Update the frames...
            t_elapsed = time.time()-t_start
            if(t_elapsed > period):
                print('Cant run in that frequency')
                print('Current freq ', 1/t_elapsed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    vid_demo(args)
