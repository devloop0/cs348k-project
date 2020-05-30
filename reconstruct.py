import argparse
from autoencoder import autoencoder
import huffman
import imageio
import numpy as np
import os
import residuals
import torch
import sys

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate trained autoencoder')
parser.add_argument('npy data folder')
parser.add_argument('input model file')
parser.add_argument('output image file')
parser.add_argument('--f', dest = 'fps', default=10)
args = vars(parser.parse_args())

data_path = args['npy data folder']
model_path = args['input model file']
output_path = args['output image file']

# Load Data
compressed = residuals.load(data_path, 'compressed_test')#[:N_FRAMES]
resids = residuals.load(data_path, 'residuals_test')#[:N_FRAMES]
N, C_in, H_in, W_in = resids.shape
compressed_frames = compressed
data = torch.from_numpy(resids).float()

# Load Model
model = autoencoder(L=3, C=32, C_in=C_in).float()
print(model)
#model.load_state_dict(torch.load(model_path))
state_dict = torch.load(model_path)
encoder = model.encoder.load_state_dict(state_dict['encoder'])
decoder = model.decoder.load_state_dict(state_dict['decoder'])

def save_img(idx, frame, tag, resid=False):
    offset = 0
    if resid:
        offset = 127

    img = frame
    img = np.clip(img, 0, 1)
    img = (img * 255 + offset).astype(np.uint8)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    imageio.imwrite('{path}/{dir}/{n:010d}.png'.format(path=output_path, dir=tag, n=idx), img)
    
for dirname in ['result', 'compressed', 'reference', 'input', 'output']:
    path = os.path.join(output_path, dirname)
    if not os.path.exists(path):
        os.mkdir(path)

# Save results
total_added_bitrate = 0
for i in range(len(resids)):
    if i > 0 and i % 10 == 0:
        print('Running frame', i)
        print('Average added bitrate so far:', total_added_bitrate / i)

    # Encode
    encoded = model.encoder(data[i:i+1])

    # Grab size
    _, _, added_bitrate = huffman.encode(encoded, fps=int(args['fps']))
    total_added_bitrate += added_bitrate

    # Decode
    decoded = model.decoder(encoded)
    decoded = decoded.data.numpy()

    save_img(i, compressed[i] + decoded[0], 'result')
    save_img(i, compressed[i], 'compressed')
    save_img(i, compressed[i] + resids[i], 'reference')    
    save_img(i, resids[i], 'input', resid=True)
    save_img(i, decoded[0], 'output', resid=True) 

avg_added_bitrate = total_added_bitrate / len(resids)
with open(os.path.join(output_path, 'avg_added_bitrate.txt'), 'w') as f:
    f.write(str(avg_added_bitrate))

print('Saved results to', output_path)
