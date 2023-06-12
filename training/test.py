#Necessary built-in modules
import random
import os
import numpy as np
import sys
import os
#Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
#PIL modules
from PIL import Image
import hoop_dataset
import encoder
import cv2
import torchvision

def visualize_sample(image, pred_mask, mask, i):
    image_numpy     = image.detach().cpu().numpy().transpose(1, 2, 0)
    pred_mask_numpy = pred_mask.detach().cpu().numpy().transpose(1, 2, 0)
    mask_numpy      = mask     .detach().cpu().numpy().transpose(1, 2, 0)

    pred_mask_numpy = cv2.cvtColor(pred_mask_numpy, cv2.COLOR_GRAY2BGR)
    mask_numpy      = cv2.cvtColor(mask_numpy, cv2.COLOR_GRAY2BGR)

    vis_stack = np.hstack((image_numpy, pred_mask_numpy, mask_numpy))
    vis_stack = cv2.resize(vis_stack, (int(vis_stack.shape[1]/2), int(vis_stack.shape[0]/2)))

    cv2.imshow('sample ' + str(i), vis_stack)
    cv2.waitKey(1)

device = 'cpu'
pin_memory = False
if torch.cuda.is_available():
    device = 'cuda'
    pin_memory = True

# Create dataset
dataset = hoop_dataset.HoopDataset("../tests/hoops", "../assets")

# Set batch size
batch_size = 1

# Create data loader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

# Set up autoencoder and optimizer
autoencoder = encoder.AutoEncoder(in_channels=3, out_channels=1).to(device)

optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

# Train autoencoder
num_epochs = 500
steps_per_print = 1
for epoch in range(num_epochs):
    for i, (real_i, image, mask) in enumerate(dataloader):
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        pred_mask = autoencoder(image)

        image     = torchvision.transforms.functional.crop(image,     top=20, left=20, height=512-40, width=640-40)
        pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
        mask      = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
        weight = (mask * 10) + 1

        loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)

        loss.backward()
        optimizer.step()

        if i % steps_per_print == 0:
            visualize_sample(image[0], pred_mask[0], mask[0], real_i[0])
            print("Epoch [{}/{}], Step [{}/{}], Loss: {}".format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))

cv2.waitKey(0)
