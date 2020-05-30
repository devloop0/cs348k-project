import sys
import imageio
import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np

if len(sys.argv) != 3:
        print('Expected arguments: <ref image> <test image>');
        exit()

ref = imageio.imread(sys.argv[1])
ours = imageio.imread(sys.argv[2])
assert (ref.shape == ours.shape)

ours = np.array(ours)
ref = np.array(ref)

print(ssim(ours, ref, multichannel=True, gaussian_weights=True, data_range=255))
