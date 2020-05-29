from autoencoder import autoencoder
import imageio
import numpy as np
import residuals
import torch
import sys

if len(sys.argv) != 4:
    print('Expected arguments: <npy data folder> <input model file> <output image folder>')
    exit()
data_path, model_path, output_path = sys.argv[1:]

N_FRAMES = 1

# Load Data
compressed = residuals.load(data_path, 'compressed')[:N_FRAMES]
#reference = residuals.load(data_path, 'reference')
resids = residuals.load(data_path, 'residuals')[:N_FRAMES]
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

# Run Model
output = model(data).data.numpy()

def save_img(idx, frame, tag, resid=False):
    offset = 0
    if resid:
        offset = 127

    img = frame
    img = np.clip(img, 0, 1)
    img = (img * 255 + offset).astype(np.uint8)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    imageio.imwrite(output_path + '/%010d_'%idx + tag + '.png', img)

# Save results
for i in range(N_FRAMES):
    save_img(i, compressed[i] + output[i], 'result')
    save_img(i, compressed[i], 'compressed')
    #save_img(i, reference[i], 'reference')
    save_img(i, compressed[i] + resids[i], 'compressed+input')    
    save_img(i, resids[i], 'input', resid=True)
    save_img(i, output[i], 'output', resid=True) 
       
print('Saved results to', output_path)
