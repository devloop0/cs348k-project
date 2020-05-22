import torch
import imageio
import glob
import os
import numpy as np

def calc(ref_path, image_path):
    residuals, our_frames, ref_frames = [], [], []
    for curr_path in glob.glob(image_path + '/*.png'):
        file_name = os.path.basename(curr_path)

        ours = imageio.imread(image_path + '/' + file_name)
        ref = imageio.imread(ref_path + '/' + file_name)

        assert(len(ours.shape) == 3)
        assert(len(ref.shape) == 3)
        assert(ours.shape == ref.shape)

        ours = np.swapaxes(ours, 0, 2)
        ours = np.swapaxes(ours, 1, 2)
        ref = np.swapaxes(ref, 0, 2)
        ref = np.swapaxes(ref, 1, 2)
        ours = ours.astype(np.float32) / 255.
        ref = ref.astype(np.float32) / 255.

        our_frames.append(ours)
        ref_frames.append(ref)
        residuals.append(ours - ref)

    return torch.from_numpy(np.array(our_frames)), \
            torch.from_numpy(np.array(ref_frames)), \
            torch.from_numpy(np.array(residuals))
