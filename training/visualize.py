import random
import os
import numpy as np
import sys
import os
import datetime
import torch
import hoop_dataset
import encoder
import cv2
import torchvision
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

#Initialize writer
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

#Check if GPU is available and set device and pin memory access accordingly
device = 'cpu'
pin_memory = False
if torch.cuda.is_available():
    device = 'cuda'
    pin_memory = True

# Create a training set that uses specified quantity of the images in dataset and a validation set
# that uses remaining images in the dataset
dataset = hoop_dataset.HoopDataset("../assets/hoops", "../assets/training_data")
dataset_size = len(dataset)
TRAINSET_SIZE = int(0.8*dataset_size)
VALIDATIONSET_SIZE = dataset_size - TRAINSET_SIZE

sets = random_split(dataset, [TRAINSET_SIZE, VALIDATIONSET_SIZE])
trainset = sets[0]
validation_set = sets[1]

# Create a dataloader with batch size 1
BATCH_SIZE = 1
dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
validateloader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

# Set up autoencoder and optimizer
autoencoder = encoder.AutoEncoder(in_channels=3, out_channels=1).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

#Return an image tensor showing three images
#Left image: Original image with hoop
#Middle image: Model prediction of where the hoop is
#Right image: Actual location of where the hoop is
def visualize_sample(image, pred_mask, mask, i):
    image_numpy = image.detach().cpu().numpy().transpose(1, 2, 0)
    pred_mask_numpy = pred_mask.detach().cpu().numpy().transpose(1, 2, 0)
    mask_numpy = mask.detach().cpu().numpy().transpose(1, 2, 0)

    pred_mask_numpy = cv2.cvtColor(pred_mask_numpy, cv2.COLOR_GRAY2BGR)
    mask_numpy = cv2.cvtColor(mask_numpy, cv2.COLOR_GRAY2BGR)

    vis_stack = np.hstack((image_numpy, pred_mask_numpy, mask_numpy))
    vis_stack = cv2.resize(vis_stack, (int(vis_stack.shape[1] / 2), int(vis_stack.shape[0] / 2)))

    tensor = torch.from_numpy(vis_stack.transpose(2, 0, 1))
    return(tensor)

NUM_EPOCHS = int(input("How many epochs do you want to run for (Recommended number is 750): "))
IMG_DISPLAY_FREQ = 100
for epoch in range(NUM_EPOCHS):
    #Training loop
    for i, (real_i, image, mask) in enumerate(dataloader):      
        #Clear gradients and obtain predicted mask of current image
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        pred_mask = autoencoder(image)

        #Crop image appropriately
        image = torchvision.transforms.functional.crop(image,     top=20, left=20, height=512-40, width=640-40)
        pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
        mask = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
        weight = (mask * 10) + 1
        
        #Compute binary cross entropy loss between prediction and target, perform backpropogation,
        #and update model parameters appropriately.
        loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)
        loss.backward()
        optimizer.step()

        #Display images in interval determined by variable IMG_DISPLAY_FREQ
        if((epoch * len(dataloader) + i) % IMG_DISPLAY_FREQ == 0):
                writer.add_image('Sample ' + str(epoch*len(dataloader) + i), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')
        
        #Write binary cross entropy loss between prediction and target to logs
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

    #Validation loop
    total_loss = 0
    curr_display = 0
    for i, (real_i, image, mask) in enumerate(validateloader):
        #Clear gradients and obtain predicted mask of current image
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        pred_mask = autoencoder(image)

        #Crop image appropriately
        image = torchvision.transforms.functional.crop(image,     top=20, left=20, height=512-40, width=640-40)
        pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
        mask = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
        weight = (mask * 10) + 1
        
        #Add binary cross entropy loss of item to total loss of all items
        loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)
        total_loss += loss

        #Display the first image in the validation set until its loss is below 0.1.
        #Switch to next when loss goes below 0.05
        if(i == curr_display):
            writer.add_image('Test ' + str(epoch), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')
            if(loss < 0.05):
                 curr_display = (curr_display + 1) % len(validation_set)
    #Write average loss of items in validation set to logs
    writer.add_scalar('Loss/Average Validation', total_loss/len(validateloader), epoch)
writer.close()