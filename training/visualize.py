# import random
# import os
# import numpy as np
# import sys
# import os
# import datetime
# import torch
# import hoop_dataset
# import encoder
# import cv2
# import torchvision
# import matplotlib.pyplot as plt
# from torch import nn, optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms, models
# from torch.utils.data import Dataset, DataLoader, random_split
# import tensorflow as tf
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# from test_video import test
# #Initialize writer
# log_dir = "/media/tanujthakkar/EVIMO2_v1/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir=log_dir)

# #Check if GPU is available and set device and pin memory access accordingly
# device = 'cpu'
# pin_memory = False
# if torch.cuda.is_available():
#     device = 'cuda'
#     pin_memory = True

# # Create a training set that uses specified quantity of the images in dataset and a validation set
# # that uses remaining images in the dataset
# dataset = hoop_dataset.HoopDataset("../assets/hoops", "../assets/training_data2")
# prelabeled_dataset = hoop_dataset.HoopDataset("../assets/masks", "../assets/backgrounds_with_hoops", preset=True)
# dataset_size = len(dataset)
# TRAINSET_SIZE = int(0.8*dataset_size)
# VALIDATIONSET_SIZE = dataset_size - TRAINSET_SIZE

# sets = random_split(dataset, [TRAINSET_SIZE, VALIDATIONSET_SIZE])
# trainset = sets[0]
# validation_set = sets[1]

# # Create a dataloader with batch size 1
# BATCH_SIZE = 1
# dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
# validateloader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
# prelabeledloader = DataLoader(prelabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
# print(prelabeledloader)
# print(validateloader)
# # Set up autoencoder and optimizer
# autoencoder = encoder.AutoEncoder(in_channels=3, out_channels=1).to(device)
# optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

# #Return an image tensor showing three images
# #Left image: Original image with hoop
# #Middle image: Model prediction of where the hoop is
# #Right image: Actual location of where the hoop is
# def visualize_sample(image, pred_mask, mask, i):
#     print(image.shape)
#     print(pred_mask.shape)
#     print(mask.shape)
#     image_numpy = image.detach().cpu().numpy()#.transpose(1, 2, 0)
#     pred_mask_numpy = pred_mask.detach().cpu().numpy()#.transpose(1, 2, 0)
#     mask_numpy = mask.detach().cpu().numpy()#.transpose(1, 2, 0)

#     pred_mask_numpy = cv2.cvtColor(pred_mask_numpy, cv2.COLOR_GRAY2BGR)
#     mask_numpy = cv2.cvtColor(mask_numpy, cv2.COLOR_GRAY2BGR)

#     vis_stack = np.hstack((image_numpy, pred_mask_numpy, mask_numpy))
#     vis_stack = cv2.resize(vis_stack, (int(vis_stack.shape[1] / 2), int(vis_stack.shape[0] / 2)))

#     tensor = torch.from_numpy(vis_stack.transpose(2, 0, 1))
#     return(tensor)

# model_path = "./saved_model"
# NUM_EPOCHS = int(input("How many epochs do you want to run for (Recommended number is 750): "))
# min_loss = 0.1
# dataloader_length = len(dataloader)
# for epoch in range(NUM_EPOCHS):
#     #Training loop
#     for i, (real_i, image, mask) in enumerate(dataloader):    
#         try:  
#             #Clear gradients and obtain predicted mask of current image
#             image = image.to(device)
#             mask = mask.to(device)
#             optimizer.zero_grad()
#             pred_mask = autoencoder(image)

#             #Crop image appropriately
#             image = torchvision.transforms.functional.crop(image,     top=20, left=20, height=512-40, width=640-40)
#             pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
#             mask = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
#             weight = (mask * 10) + 1
            
#             #Compute binary cross entropy loss between prediction and target, perform backpropogation,
#             #and update model parameters appropriately.
#             loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)
#             loss.backward()
#             optimizer.step()
            
#             #Write binary cross entropy loss between prediction and target to logs
#             writer.add_scalar('Loss/train', loss.item(), epoch * dataloader_length + i)
#             print((epoch, dataloader_length, i,  epoch * dataloader_length + i))
#         except:
#              print("Error")

#     #Validation loop
#     total_loss = 0
#     validate_len = 0
#     for i, (real_i, image, mask) in enumerate(validateloader):
#         try:
#             print("Validating" + str(i))
#             #Clear gradients and obtain predicted mask of current image
#             image = image.to(device)
#             mask = mask.to(device)
#             optimizer.zero_grad()
#             pred_mask = autoencoder(image)
            
#             #Crop image appropriately
#             image = torchvision.transforms.functional.crop(image,     top=21, left=20, height=512-40, width=640-40)
#             pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
#             mask = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
#             weight = (mask * 10) + 1
            
#             #Add binary cross entropy loss of item to total loss of all items
#             loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)
#             total_loss += loss.detach().cpu()

#             #Display the first image in the validation set until its loss is below 0.1.
#             #Switch to next when loss goes below 0.05
#             if(i % max(1,((epoch % 3) * 100)) == 0):
#                 if(total_loss/len(validateloader) < min_loss):
#                     torch.save(autoencoder.state_dict(), model_path)
#                     min_loss = total_loss/len(validateloader)
#             validate_len += 1
#         except:
#              print("error")
#         writer.add_scalar('Loss/Average Validation', total_loss/validate_len, epoch)
#         writer.add_image('Testing ' + str(epoch) + ' ' + str(i), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')
#     for i, (real_i, image, mask) in enumerate(prelabeledloader):
#             print((real_i, image, mask))
#             print("Testing " + str(i))
#             #Clear gradients and obtain predicted mask of current image
#             image = image[0].to(device)
#             mask = mask.to(device)
#             optimizer.zero_grad()
#             pred_mask = autoencoder(image)
            
#             #Crop image appropriately
#             image = torchvision.transforms.functional.crop(image,     top=21, left=20, height=512-40, width=640-40)
#             pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
#             mask = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
#             weight = (mask * 10) + 1
#             #writer.add_image('Testing ' + str(epoch) + ' ' + str(i), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')
#     #Write average loss of items in validation set to logs
# writer.close()

# torch.save(autoencoder.state_dict(), model_path)
import random
import time
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
from test_video import test
from numba import jit
#Initialize writer
log_dir = "/media/tanujthakkar/EVIMO2_v1/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
prelabeled_dataset = hoop_dataset.HoopDataset("../assets/masks", "../assets/backgrounds_with_hoops", preset=True)
dataset_size = len(dataset)
TRAINSET_SIZE = int(0.8*dataset_size)
VALIDATIONSET_SIZE = dataset_size - TRAINSET_SIZE

sets = random_split(dataset, [TRAINSET_SIZE, VALIDATIONSET_SIZE])
trainset = torch.utils.data.ConcatDataset([sets[0], prelabeled_dataset])
validation_set = sets[1]

# Create a dataloader with batch size 1
BATCH_SIZE = 1
dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
validateloader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
prelabeledloader = DataLoader(prelabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
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

model_path = "./saved_model"
NUM_EPOCHS = int(input("How many epochs do you want to run for (Recommended number is 750): "))

IMG_DISPLAY_FREQ = 1000
min_loss = 0.10

dataloader_length = len(dataloader)

for epoch in range(NUM_EPOCHS):
    #Training loop
    main_start = time.perf_counter()
    is_first = True
    for i, (real_i, image, mask) in enumerate(dataloader):    
        try:
            #Clear gradients and obtain predicted mask of current image
            start_time = time.perf_counter()
            image = image.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred_mask = autoencoder(image)
            end_time = time.perf_counter()
            masking_time = end_time - start_time
            #Crop image appropriately
            start_time = time.perf_counter()
            crop_transform = lambda im : torchvision.transforms.functional.crop(im, top=20, left=20, height=512-40, width=640-40)
            image = crop_transform(image)
            pred_mask = crop_transform(pred_mask)
            mask = crop_transform(mask)
            weight = (mask * 10) + 1
            end_time = time.perf_counter()
            cropping_time = end_time - start_time
            #Compute binary cross entropy loss between prediction and target, perform backpropogation,
            #and update model parameters appropriately.
            start_time = time.perf_counter()
            loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)
            loss.backward()
            optimizer.step()
            end_time = time.perf_counter()
            update_time = end_time - start_time
            #Display images in interval determined by variable IMG_DISPLAY_FREQ
            # if((epoch * len(dataloader) + i) % IMG_DISPLAY_FREQ == 0):
                    #writer.add_image('Sample ' + str(epoch*len(dataloader) + i), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')
                    
            #Write binary cross entropy loss between prediction and target to logs
            #writer.add_image('Test Training ' + str(i), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')
            writer.add_scalar('Loss/train', loss.item(), epoch * dataloader_length + i)
            main_end = time.perf_counter()
            #print((epoch, dataloader_length, i,  "Masking time: " + str(masking_time), "Cropping Time: " + str(cropping_time), "Update Time: " + str(update_time), "Total: " + str(main_end - main_start)))
        except:
             print("Error")

    #Validation loop
    total_loss = 0
    validate_len = 0
    for i, (real_i, image, mask) in enumerate(validateloader):
        try:
            print("Validating" + str(i))
            #Clear gradients and obtain predicted mask of current image
            image = image.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred_mask = autoencoder(image)

            #Crop image appropriately
            image = torchvision.transforms.functional.crop(image,     top=21, left=20, height=512-40, width=640-40)
            pred_mask = torchvision.transforms.functional.crop(pred_mask, top=20, left=20, height=512-40, width=640-40)
            mask = torchvision.transforms.functional.crop(mask,      top=20, left=20, height=512-40, width=640-40)
            weight = (mask * 10) + 1
            
            #Add binary cross entropy loss of item to total loss of all items
            loss = F.binary_cross_entropy(pred_mask, mask, weight=weight)
            total_loss += loss.detach().cpu()

            #Display the first image in the validation set until its loss is below 0.1.
            #Switch to next when loss goes below 0.05
            if(i % 10000 == 0):
                writer.add_image('Validation ' + str(epoch) + ' ' + str(i), visualize_sample(image[0], pred_mask[0], mask[0], real_i[0]), dataformats='CHW')

            validate_len += 1
        except:
             print("error")
        writer.add_scalar('Loss/Average Validation', total_loss/validate_len, epoch)
    for i, (real_i, image, mask) in enumerate(prelabeledloader):
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        pred_mask = autoencoder(image).to(device)

        # Adjust pred_mask dimensions
        pred_mask = pred_mask.squeeze(0)

        # View image
        image_tensor = image.squeeze().cpu()
        mask_tensor = mask.squeeze().cpu()
        pred_mask_tensor = pred_mask.cpu()

        # Ensure mask_tensor and pred_mask_tensor have the same dimensions
        mask_tensor = mask_tensor[:3, :, :]
        pred_mask_tensor = pred_mask_tensor.expand(3, -1, -1)

        writer.add_image('Image ' + str(i), torch.cat((image_tensor, mask_tensor, pred_mask_tensor), 2))


        #writer.add_image("Test " + str(i), image, dataformats="HWC")
    if(total_loss/len(validateloader) < min_loss):
        torch.save(autoencoder.state_dict(), model_path)
        min_loss = total_loss/len(validateloader)
        #writer.add_image('Test_Video' + str(epoch), test())
    #Write average loss of items in validation set to logs

writer.close()


torch.save(autoencoder.state_dict(), model_path)