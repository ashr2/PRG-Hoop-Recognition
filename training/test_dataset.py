#this is the code that worked before

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
import torchvision
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def display_tensor_as_image(tensor):
    # If the tensor has a batch dimension, remove it
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
   
    # Convert tensor to numpy array
    img = tensor.permute(1, 2, 0).detach()
    
    # Display the image
    # plt.imshow(img)
    # plt.axis('off')  # Turn off axis numbers
    # plt.show()
    return(img)

#this is the new dataset
from torch.utils.data import Dataset, DataLoader
import os
import torch
import encoder
class TestDataset(Dataset): 

    def __init__(self, image_dir, model_path):
        """
        Args:
            image_dir (string): Path to directory with all images
        """
        #Directory storing backgrounds
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        #Store model path
        self.model_path = model_path

        self.cache = []
        #Remove DS_Store file in directories
        if(".DS_Store") in self.images:
            self.hoops.remove(".DS_Store")

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, idx):
        # if(not(idx in self.cache)):
        image_idx = idx % len(self.images)
        image = self.images[image_idx]
        print(self.image_dir + image)
        image = Image.open(self.image_dir + image)

        tf=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512,640)),
        ])
        tensor = tf(image)
        #tensor = tensor.unsqueeze(0).to(self.device)
            #pred_mask = self.autoencoder(tensor).to(self.device)

            # Adjust pred_mask dimensions
            #pred_mask = pred_mask.squeeze(0)

            # View image
            #image_tensor = image.squeeze().cpu()
            #pred_mask_tensor = pred_mask.cpu()

            # Ensure mask_tensor and pred_mask_tensor have the same dimension
            #pred_mask_tensor = pred_mask_tensor.expand(3, -1, -1)

            #result = torch.cat((image_tensor, pred_mask_tensor), 2)
            #image_pred = display_tensor_as_image(result)
            #plt.imsave("./test_cache/"+str(idx)+".png", image_pred)
        #result = Image.open("./test_cache/"+str(idx)+".png")
        # return 1234, 1234
        return idx, tensor
def show(img):
    display_img = img.cpu()#.permute(1,2,0)
    plt.imshow(display_img)

if __name__ == "__main__":
    dataset = TestDataset("../assets/backgrounds_with_hoops/", "./saved_model")
    pin_memory = True
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=pin_memory)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    with torch.inference_mode():
        autoencoder = encoder.AutoEncoder(in_channels=3, out_channels=1).to(device)

        #self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.0001)
        # Load the state dict previously saved
        state_dict = torch.load("./saved_model", map_location=device)
        #Load the state dict to the model
        autoencoder.load_state_dict(state_dict)

        for i, (real_i, tensor) in enumerate(dataloader):
            print(i)
            tensor = tensor.to(device)
            print(type(tensor))
            print(tensor.shape)
            #tensor = tensor.unsqueeze(0).to(device)
            pred_mask = autoencoder(tensor)
            pred_mask = pred_mask.squeeze()
            pred_mask = pred_mask.cpu()
            #print(type(pred_mask))
            #print(pred_mask.shape)
            plt.imshow(pred_mask)
            plt.axis('off')
            plt.show()
            # print(pred_mask.shape)
