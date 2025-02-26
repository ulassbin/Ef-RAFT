import sys
import os
import colorama as color
from collections import deque

color.init()
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
from utils.utils import InputPadder, InputPadder2
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

def to_tensor_npy_batch(frames):
    img = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return img.to(DEVICE)

def to_image(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    return flo

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()



def init_video(width, height, name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
    #fourcc = cv2.VideoWriter_fourcc(*'XVID') # avi
    writer = cv2.VideoWriter(name, fourcc, 
                         30, (width,height)) # Saves at 25 FPS
    return writer

def get_video_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return None, None

    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()

    return width, height

def create_directory(directory_path):
    """
    Check if a directory exists, if not, create it.

    Parameters:
        directory_path (str): The path of the directory to check/create.

    Returns:
        str: The path of the directory.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error: Failed to create directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")

    return directory_path

def saveFlow(vid, path, vid_name):
    #width, height = get_video_dimensions(file)
    if(len(vid) == 0):
        print("Error in vid size!")
        return False
    height, width, dim = vid[0].shape
    print("Vid dims ({}, {}, {})".format(width, height, dim))
    path = path +"/"+vid_name
    writer = init_video(width, height, path)
    iter=0
    for frame in vid:
        iter+=1
        writer.write(frame)
    writer.release()
    return len(vid) > 0

def create_blank_image_like(image):
    height, width, _ = image.shape
    return np.zeros((height, width, 3), dtype=np.uint8)

def processVid(args, path, model): # This produces one less frame for some reason
    flow_frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Unable to open video file: ", path)
        cap.release()
        return flow_frames, False
    ret, old_frame = cap.read() # 1 read, 1 append?
    flow_frames.append(create_blank_image_like(old_frame)) # + 1 appended here.
    while True: # create flow from, t+1 and t
        ret, frame = cap.read()
        if not ret:
            print("End of vid...")
            break
        image1 = to_tensor(old_frame) # Might require a conversion here...
        image2 = to_tensor(frame)
        padder = InputPadder(image1.shape)
        image1pad, image2pad = padder.pad(image1, image2)
        flow_low, flow_up = model(image1pad, image2pad, iters=32, test_mode=True)
        flow_frame = to_image(flow_up)
        flow_frames.append(flow_frame)
        old_frame = copy.deepcopy(frame) # Update the frames...
        #viz(image1pad, flow_up))
    print(color.Fore.BLUE + "Written {}".format(iter) + color.Style.RESET_ALL)
    cap.release()
    return flow_frames, True

def get_vid_as_npy(vid_path, padding =True):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # add a zero padding frame at start
    if padding:
        frames.insert(0, np.zeros_like(frames[0]))
    return np.array(frames)
    

def processVid_batch(args, path, model, batch_size=50):

  vid_npy = get_vid_as_npy(path)
  flow_frames = []
  data_len = len(vid_npy)
  with torch.no_grad():
      for i in range(0, data_len-1, batch_size):
          print('Progress {}%'.format(i/data_len*100))
          start_total = time.time()
          # 1️⃣ Convert frames to tensors
          start = time.time()
          image1_batch = to_tensor_npy_batch(vid_npy[i:min(i+batch_size, data_len-1)])
          image2_batch = to_tensor_npy_batch(vid_npy[i+1:min(i+1+batch_size, data_len)])
          end = time.time()
          #print(f"🕒 Tensor conversion time: {end - start:.4f} sec")

          # 2️⃣ Pad images
          start = time.time()
          padder = InputPadder2(image1_batch.shape)
          image1_pad, image2_pad = padder.pad(image1_batch, image2_batch)
          end = time.time()
          #print(f"🕒 Padding time: {end - start:.4f} sec")

          print(color.Fore.GREEN + 'Image1 {}, Image2 {}'.format(image1_pad.shape, image2_pad.shape) + color.Style.RESET_ALL)

          # 3️⃣ Move to GPU
          start = time.time()
          image1_pad = image1_pad.to("cuda")
          image2_pad = image2_pad.to("cuda")
          end = time.time()
          #print(f"🕒 GPU transfer time: {end - start:.4f} sec")

          # 4️⃣ Run model
          start = time.time()
          flow_low, flow_up = model(image1_pad, image2_pad, iters=32, test_mode=True)
          end = time.time()
          #print(f"🕒 Model inference time: {end - start:.4f} sec")

          # 5️⃣ Convert to images
          start = time.time()
          
          for j in range(image1_batch.shape[0]):  # Use different index than `i` to avoid confusion
              flow_frame = to_image(flow_up[j].unsqueeze(0).cpu())
              flow_frames.append(flow_frame)
          end = time.time()
          #print(f"🕒 Image conversion time: {end - start:.4f} sec")

          total_end = time.time()
          #print(f"🕒 Total loop iteration time: {total_end - start_total:.4f} sec")
          #print("=" * 50)  # Separator for readabilityh
  print(f"Processed {len(flow_frames)} flow frames.")
  #cap.release()
  return flow_frames, True



def get_video_length_raw(video_path): 
    # Just checks it by manually opening up frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()

    return frame_count

def get_video_length(video_path):
    """
    Get the length of a video in frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return -1

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def compare_video_lengths(video1_path, length2):
    """
    Compare the lengths of two videos.

    Parameters:
        video1_path (str): The path of the first video file.
        video2_path (str): The path of the second video file.

    Returns:
        str: A message indicating which video is longer.
    """
    length1 = get_video_length_raw(video1_path)
    print("Lengths are {}, {}".format(length1, length2))
    if length1 == -1 or length2 == -1:
        print(color.Fore.MAGENTA + "Error: Failed to get the lengths of the videos.")

    if length1 > length2:
        print(color.Fore.RED, "Video is longer than OpticalFlow.")

    elif length1 < length2:
        print(color.Fore.YELLOW + "OpticalFlow is longer than Video.")
    else:
        print(color.Fore.GREEN + "Video and OpticalFlow have the same length.")
    print(color.Style.RESET_ALL)

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    base_path = args.path # Path for video sources...

    with torch.no_grad():
        print("Base path is ", base_path)
        folder = os.fsencode(base_path)
        class_name = 'None'
        for file_item in os.listdir(folder):
            file = base_path +"/"+ os.fsdecode(file_item)
            if(os.path.isdir(os.fsencode(file))):
                print('Processing files in the dir: {}'.format(os.fsdecode(file)))
                class_name = os.fsdecode(file_item)
                save_dir = args.flow_path +"/"+class_name
                create_directory(save_dir)
                for vid_name in os.listdir(file):
                    vid_file = os.fsdecode(file) +"/"+vid_name
                    print("Opening up vid: ", vid_file)
                    frames, ret = processVid_batch(args, vid_file, model)
                    #compare_video_lengths(vid_file, len(frames))
                    if(ret):
                        saveFlow(frames, save_dir, vid_name)
                        print("RGB {}, flow {}".format(get_video_length_raw(vid_file), get_video_length_raw(save_dir +"/"+vid_name))) 
                    else:
                        print("Unable to get flow frames")
                    model.eval()


def run_batched(args, batch_size=50):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(RAFT(args)).to(device)
    model.load_state_dict(torch.load(args.model))
    
    model = model.module
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(args.path)  # or VideoCapture(0) for webcam
    freq = 10
    period = 1 / freq

    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    #batch_size = 10  # Process 4 frame pairs at a time (adjust as needed)
    frame_queue = deque(maxlen=batch_size + 1)

    with torch.no_grad():
        while True:
            t_start = time.time()

            # Read and store frames in a queue
            while len(frame_queue) < batch_size + 1:
                ret, frame = cap.read()
                if not ret:
                    print("End of video...")
                    return
                frame_queue.append(frame)

            # Prepare batch
            image1_batch = []
            image2_batch = []
            for i in range(batch_size):
                image1 = to_tensor(frame_queue[i])
                image2 = to_tensor(frame_queue[i + 1])
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                image1_batch.append(image1)
                image2_batch.append(image2)

            # Stack frames into a batch (B, C, H, W)
            image1_batch = torch.cat(image1_batch, dim=0).to(device)
            image2_batch = torch.cat(image2_batch, dim=0).to(device)

            # Run model on batch
            flow_low, flow_up = model(image1_batch, image2_batch, iters=20, test_mode=True)

            # Visualize each flow result
            #for i in range(batch_size):
            #    viz2(image1_batch[i].cpu(), flow_up[i].cpu())

            # Update frames (keep last frame in queue)
            frame_queue.popleft()

            t_elapsed = time.time() - t_start
            if t_elapsed > period:
                print('Cant run at desired frequency')
                print('Current freq:', 1 / t_elapsed)