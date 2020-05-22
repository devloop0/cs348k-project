import torch
import imageio
import glob
import os
import numpy as np
import sys
from skimage.metrics import structural_similarity as ssim

if len(sys.argv) != 3:
    print('Expected arguments: <ref path> <image path>')
    exit()

REF_PATH = sys.argv[1]
IMAGE_PATH = sys.argv[2]
# REF_PATH = '/home/ubuntu/data/2011_09_28/2011_09_28_drive_0038_sync/image_03/data/'
# IMAGE_PATH='/home/ubuntu/testing/frames/'

our_frames, ref_frames = [], []

for image_path in glob.glob(IMAGE_PATH + '*.png'):
    file_name = os.path.basename(image_path)

    ours = imageio.imread(IMAGE_PATH + file_name)
    ref = imageio.imread(REF_PATH + file_name)

    assert(len(ours.shape) == 3)
    assert(len(ref.shape) == 3)
    assert(ours.shape == ref.shape)

    # ours = np.swapaxes(ours, 0, 2)
    # ours = np.swapaxes(ours, 1, 2)
    # ref = np.swapaxes(ref, 0, 2)
    # ref = np.swapaxes(ref, 1, 2)
    # ours = ours.astype(np.float32) / 255.
    # ref = ref.astype(np.float32) / 255.

    our_frames.append(ours)
    ref_frames.append(ref)

our_frames = np.array(our_frames)
ref_frames = np.array(ref_frames)
print(ssim(our_frames, ref_frames, multichannel=True, gaussian_weights=True, data_range=255))
