import torch

class autoencoder(torch.nn.Module):
    def __init__(self, L=3, C=32, C_in=3):
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
        x = torch.clamp(x, min=-1, max=1)
        x = self.decoder(x)
        return x
                
