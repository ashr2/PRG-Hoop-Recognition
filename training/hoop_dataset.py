#Built-in modules
import cv2
import numpy as np
import random
import os
import sys
from joblib import Memory
#Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
#Image modules
import PIL
from PIL import Image

def add_hoop_to_background(background_image, hoop_image):
    max_hoop_size = (background_image.size[0] - 1, background_image.size[1] - 1)
    hoop_image.thumbnail(max_hoop_size, PIL.Image.ANTIALIAS)

    val = min(background_image.size[1], background_image.size[0])//2
    size = random.randint(min(100, val),max(100, val))
    maxsize = (min(size, hoop_image.size[1]), min(size, hoop_image.size[1]))
    hoop_image.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    
    #Paste hoop onto background
    x_center = random.randint(0, background_image.size[0] - hoop_image.size[0])
    y_center = random.randint(0, background_image.size[1] - hoop_image.size[1])
    #im.paste(mouse, (40,40), mouse)
    background_image.paste(hoop_image, (x_center, y_center), hoop_image)

    #Paste hoop onto black background
    black_image = Image.new("L", background_image.size)
    hoop_mask = hoop_image.convert("L")
    black_image.paste(hoop_mask, (x_center, y_center))
    black_image = black_image.crop((0, 0, background_image.size[0], background_image.size[1]))
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
cache_dir = "/media/tanujthakkar/EVIMO2_v1"

memory = Memory(cache_dir, verbose=0)

class HoopDataset(Dataset): 

    def __init__(self, hoop_dir, background_dir, preset = False, transform=None):
        """
        Args:
            hoop_dir (string): Path to directory with all hoops
            background_dir (string): Path to directory with all backgrounds
        """
        self.preset = preset
        self.background_dir = background_dir
        self.hoop_dir = hoop_dir
        self.hoops = sorted(os.listdir(hoop_dir))
        if(".DS_Store") in self.hoops:
            self.hoops.remove(".DS_Store")
        self.backgrounds = sorted(os.listdir(background_dir))
        if(".DS_Store" in self.backgrounds):
            self.backgrounds.remove(".DS_Store")
        self.num_backgrounds = len(self.backgrounds)
        self.transform = transform
        self.cached_getitem = memory.cache(self._getitem)

    def __len__(self):
        return(self.num_backgrounds * len(self.hoops))


    def _getitem(self, idx):
        hoop_idx = idx % len(self.hoops)
        back_idx = int(idx / len(self.hoops)) % self.num_backgrounds
        if self.preset == False:
            hoop = Image.open(self.hoop_dir + "/" + self.hoops[hoop_idx])
            background = Image.open(self.background_dir + "/" + self.backgrounds[back_idx])
            hoop_back, hoop_black = add_hoop_to_background(background, hoop)
            
            tf=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512,640)),
            ])
            hoop_back  = tf(hoop_back)
            hoop_black = tf(hoop_black)
            hoop_black = (hoop_black > 0).type(torch.float)

            data = (hoop_back, hoop_black)

            return idx, *data
        else:
            image = Image.open(self.background_dir + "/" + "background" + str(hoop_idx) + ".png")
            image = image.convert('RGB')
            mask = Image.open(self.hoop_dir + "/" + "mask" + str(hoop_idx) + ".png")
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 640))
            ])
            mask = tf(mask)
            mask = (mask > 0).type(torch.float)
            data = (tf(image), mask)
            return idx, *data

    def __getitem__(self, idx):
        return self.cached_getitem(idx)
