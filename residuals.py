import torch
import imageio
import glob
import os
import numpy as np
import sys

def calc(ref_path, image_path, which):
    paths = list(glob.glob(image_path + '/*.png'))
    file_name = os.path.basename(paths[0])
    frame = imageio.imread(image_path + '/' + file_name)
    shape = frame.shape

    result = np.empty((len(paths), shape[2], shape[0], shape[1]), dtype=np.float32)
    
    for i, curr_path in enumerate(paths):
        if i % 100 == 0:
            print('Loading', which, 'image', i)
            
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

        if which == 'reference':
            result[i] = ref
        elif which == 'compressed':
            result[i] = ours
        elif which == 'residuals':
            result[i] = ref - ours
        else:
            raise RuntimeError('Invalid type "' + which + '"')
               
    return result

def load(output_path, which):
    with open(output_path + '/' + which +'.npy', 'rb') as f:
        return np.load(f)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Expected arguments: <ref images folder> <compressed images folder> <output data folder> <which>')
        exit()

    ref_path, compressed_path, output_path, which = sys.argv[1:]
    result = calc(ref_path, compressed_path, which)

    np.random.seed(348)
    rand = np.random.rand(len(result))
    # print(rand)
    train_keys = rand <= 0.8
    test_keys = rand > 0.8
        
    with open(output_path + '/' + which + '_train.npy', 'wb') as f:
        np.save(f, result[train_keys])

    with open(output_path + '/' + which + '_test.npy', 'wb') as f:
        np.save(f, result[test_keys])
    
    
