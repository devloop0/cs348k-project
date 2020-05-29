from autoencoder import autoencoder
import numpy as np
import residuals
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

if len(sys.argv) != 3:
    print('Expected arguments: <npy data folder> <output model file>')
    exit()
data_path, model_path = sys.argv[1:]

# Hyperparameters
num_epochs = 50
batch_size = 10
learning_rate = 1e-3

# Load Data
resids = torch.from_numpy(residuals.load(data_path, 'residuals')).float()
data_loader = DataLoader(resids, batch_size=batch_size, shuffle=True)
N, C_in, H_in, W_in = resids.shape

# Construct model
model = autoencoder(L=3, C=32, C_in=C_in).float()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Train
for epoch in range(num_epochs):

    if epoch > 0 and epoch % 5 == 0:
        learning_rate /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print('Learning rate:', learning_rate)
        
    for data in data_loader:
        img = data
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        print('Loss:', loss)
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))

    if (epoch > 0 and epoch % 10 == 0) or epoch == num_epochs-1:
        torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict()},
               model_path)
        print('Saved model to', model_path)
        
