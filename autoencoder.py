import numpy as np
import residuals
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

if sys.argv != 3:
    print('Expected arguments: <ref path> <image path> <output path>')
    exit()
ref_path, image_path, output_path = sys.argv[1:]

# Hyperparameters
num_epochs = 1
batch_size = 10
learning_rate = 1e-3

# Load Data
compressed, refrence, resids = residuals.calc(ref_path, image_path)
N, C_in, H_in, W_in = resids.shape
data_loader = DataLoader(resids, batch_size=batch_size, shuffle=True)

class autoencoder(torch.nn.Module):
    def __init__(self, L=3, C=32):
        super(autoencoder, self).__init__()

        kernel_size = 3

        encoder_layers = []
        decoder_layers = []

        # Encoder
        for l in range(L):
            C0, C1 = C, C
            if l == 0:
                C0 = C_in

            encoder_layers.append(torch.nn.Conv2d(C0, C1, kernel_size, stride=2, padding=1))
            encoder_layers.append(torch.nn.BatchNorm2d(C))

            if l != L-1: # No ReLU on last iteration
                encoder_layers.append(torch.nn.ReLU())

        # Binarizer
        encoder_layers.append(torch.nn.Hardtanh())

        # Decoder
        for l in range(L):
            C0, C1 = C, 4*C
            if l == L-1:
                C1 = C_in*4

            decoder_layers.append(torch.nn.Conv2d(C0, C1, kernel_size, stride=1, padding=1))
            decoder_layers.append(torch.nn.PixelShuffle(2))

            if l != L-1: # No Batch + ReLU on last iteration
                decoder_layers.append(torch.nn.BatchNorm2d(C))
                decoder_layers.append(torch.nn.ReLU())


        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        

model = autoencoder(L=2)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # TODO: weight decay
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
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

torch.save(model.state_dict(), output_path)
        
