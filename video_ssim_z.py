import torch
import imageio
import glob
import os
import numpy as np
import sys
from skimage.metrics import structural_similarity as ssim

if len(sys.argv) != 3:
    print('Expected arguments: <folder 1> <folder 2>')
    exit()

PATH_1 = sys.argv[1]
PATH_2 = sys.argv[2]

total = 0
count = 0
for image_path in glob.glob(PATH_1 + '*.png'):
    file_name = os.path.basename(image_path)

    img_1 = imageio.imread(PATH_1 + file_name)
    img_2 = imageio.imread(PATH_2 + file_name)

    assert(len(img_1.shape) == 3)
    assert(len(img_2.shape) == 3)
    assert(img_1.shape == img_2.shape)

    score = ssim(img_1, img_2, multichannel=True, gaussian_weights=True, data_range=255)
    total += score
    count += 1

    if count % 10 == 0:
        print('Average SSIM after', count, 'images:', total/count)

print('Average SSIM:', total/count)
