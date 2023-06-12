import torch
import torchvision
import torchvision.transforms as transforms
import hoop_dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Original 
        self.encoder_conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, 
                      kernel_size=(16, 16), stride=2)
        self.encoder_conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, 
                      kernel_size=(8, 8), stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=1, return_indices=True)
        # self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=1, padding=1)
        self.decoder_conv2d_1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                               kernel_size=(8, 8), stride=2)
        self.decoder_conv2d_2 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channels,
                               kernel_size=(16, 16), stride=2)

        # torch.nn.init.normal_(self.encoder_conv2d_1.weight)
        # torch.nn.init.normal_(self.encoder_conv2d_2.weight)
        # torch.nn.init.normal_(self.decoder_conv2d_1.weight)
        # torch.nn.init.normal_(self.decoder_conv2d_2.weight)

    def forward(self, x):
        # Original
        encoded_conv2d_1 = self.encoder_conv2d_1(x)
        encoded_conv2d_2 = self.encoder_conv2d_2(F.relu(encoded_conv2d_1))
        # encoded, indices = self.pool(F.relu(encoded_conv2d_2))
        # decoded_unpool = self.unpool(F.relu(encoded), indices, output_size=encoded_conv2d_2.size())
        decoded_conv2d_1 = self.decoder_conv2d_1(F.relu(encoded_conv2d_2), output_size=encoded_conv2d_1.size())
        decoded = self.decoder_conv2d_2(F.relu(decoded_conv2d_1), output_size=x.size())

        # Use sigmoid to keep output between 0 and 1
        decoded = torch.sigmoid(decoded)

        return decoded