#Built-in modules
import cv2
import numpy as np
import random
import os
import sys
#Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset, DataLoader
#Image modules
import PIL
from PIL import Image
import PIL.PngImagePlugin
import time

def add_hoop_to_background(background_image, hoop_image, mask):
    BACKGROUND_WIDTH, BACKGROUND_HEIGHT = background_image.size
    NEW_BACKGROUND_WIDTH = 640
    NEW_BACKGROUND_HEIGHT = max(1, int(BACKGROUND_HEIGHT * (NEW_BACKGROUND_WIDTH/BACKGROUND_WIDTH)))
    HOOP_WIDTH, HOOP_HEIGHT = hoop_image.size
    
    #transform hoop
    HOOP_ROTATION = random.randint(0, 360)
    hoop_image = hoop_image.rotate(HOOP_ROTATION, PIL.Image.NEAREST, expand=1)
    mask = mask.rotate(HOOP_ROTATION, PIL.Image.NEAREST, expand=1)
    #Resize background
    background_image = background_image.resize((NEW_BACKGROUND_WIDTH, NEW_BACKGROUND_HEIGHT))

    X_CENTER = random.randint(0, max(NEW_BACKGROUND_WIDTH - HOOP_WIDTH, NEW_BACKGROUND_WIDTH//2))
    Y_CENTER = random.randint(0, max(NEW_BACKGROUND_HEIGHT - HOOP_HEIGHT, NEW_BACKGROUND_HEIGHT//2))
    #Paste hoop onto background
    #im.paste(mouse, (40,40), mouse)
    background_image.paste(hoop_image, (X_CENTER, Y_CENTER), hoop_image)

    #Paste hoop onto black background
    black_image = Image.new("L", (NEW_BACKGROUND_WIDTH, NEW_BACKGROUND_HEIGHT))
    black_image.paste(mask, (X_CENTER, Y_CENTER))
    return background_image, black_image

def transform_image(orig_img):
    transformed_img = orig_img
    #Rotate image
    rotater = T.RandomRotation(degrees=(0, 180))
    rotated_img = [rotater(transformed_img)]
    transformed_img = rotated_img[0]

    #Flip vertically
    vflipper = T.RandomVerticalFlip(p=0.5)
    transformed_imgs = [vflipper(transformed_img)]
    transformed_img = transformed_imgs[0]

    #Flip horizontally
    hflipper = T.RandomHorizontalFlip(p=0.5)
    transformed_imgs = [hflipper(transformed_img)]
    transformed_img = transformed_imgs[0]

    #Perspective image
    perspective_transformer = T.RandomPerspective(distortion_scale=0.8, p=1.0) #Test distortion scale, play around with it
    perspective_imgs = [perspective_transformer(transformed_img)]
    transformed_img = perspective_imgs[0]
    
    return(transformed_img)

class HoopDataset(Dataset): 

    def __init__(self, hoop_dir, background_dir, mask_dir, preset = False, transform=None):
        """
        Args:
            hoop_dir (string): Path to directory with all hoops
            background_dir (string): Path to directory with all backgrounds
        """
        self.preset = preset
        #Directory storing backgrounds
        self.background_dir = background_dir
        self.backgrounds = sorted(os.listdir(background_dir))
        #Directory storing hoops
        self.hoop_dir = hoop_dir
        #Directory storing masks
        self.mask_dir = mask_dir
        #Get list of all hoops in hoop directory
        self.hoops = sorted(os.listdir(hoop_dir))
        #Get list of all masks in mask directory
        self.masks = sorted(os.listdir(mask_dir))
        #Remove DS_Store file in directories
        if(".DS_Store") in self.hoops:
            self.hoops.remove(".DS_Store")
        
        if(".DS_Store" in self.backgrounds):
            self.backgrounds.remove(".DS_Store")

        if(".DS_Store" in self.masks):
            self.masks.remove(".DS_Store")


        self.num_backgrounds = len(self.backgrounds)
        self.transform = transform
        #Image.init()
        #Image.register_extensions("png", extensions=PIL.PngImagePlugin)

    def __len__(self):
        return(self.num_backgrounds)

    def __getitem__(self, idx):
        hoop_idx = idx % len(self.hoops)
        back_idx = idx % self.num_backgrounds
        if self.preset == False:
            #Load appropriate hoop and background based on idx
            hoop = PIL.Image.open(self.hoop_dir + "/" + self.hoops[hoop_idx], formats=['PNG'])
            background = Image.open(self.background_dir + "/" + self.backgrounds[back_idx])
            mask = Image.open(self.mask_dir + "/" + self.masks[hoop_idx], formats=['PNG'])
            #Recolor hoop
            hoop_rgb, hoop_a = hoop.split()[0:3], hoop.split()[3]
            transform = transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3)
            transformed_rgb = transform(Image.merge('RGB', hoop_rgb))
            hoop = Image.merge('RGBA', (*transformed_rgb.split(), hoop_a))

            #Get image of hoop pasted on background and of mask
            hoop_back, hoop_black = add_hoop_to_background(background, hoop, mask)

            tf=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512,640)),
            ])
            hoop_back  = tf(hoop_back)
            hoop_black = tf(hoop_black)
            hoop_black = (hoop_black > 0).type(torch.float)
            #print(" Total Time: " + str(total_end- total_start), " Alpha Time: " + str(alpha_time) + " Transform time: " + str(tran_time + to_pil_time))
            # print("Total Time: " + str(total_end - total_start))
            # max_time = max(alpha_time, tran_time + to_pil_time, pasting_time) 
            # if(max_time == alpha_time):
            #     print("Alpha")
            # elif(max_time == tran_time + to_pil_time):
            #     print("Transform")
            # elif(max_time == pasting_time):
            #     print("Pasting")
            data = (hoop_back, hoop_black)

            return idx, *data
        else:
            hoop = PIL.Image.open(self.hoop_dir + "/" + self.hoops[hoop_idx], formats=['PNG'])
            background = Image.open(self.background_dir + "/" + self.backgrounds[back_idx])
            mask = Image.open(self.mask_dir + "/" + self.masks[hoop_idx], formats=['PNG'])
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 640))
            ])
            mask = tf(mask)
            mask = (mask > 0).type(torch.float)
            data = (tf(background), mask)
            return idx, *data