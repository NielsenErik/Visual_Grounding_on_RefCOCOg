import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from printCalls import debugging, info
from PIL import Image, ImageFilter, ImageOps
import clip
import random
from torchvision import transforms

class DataAugmentation():
    
    
    def blur(img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),            
        ])
        img = transform(img)
        return img
    
    def rotate(img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
    def grayscale(img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
    def colorrand(img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.01*random.randrange(1,50), contrast=0.01*random.randrange(1,50), saturation=0.01*random.randrange(1,50), hue=0.01*random.randrange(1,50)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
    def random_crop(img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
        
    def random_augmentation(img):
        n = random.randint(0, 5)
        if n == 0:
            return DataAugmentation.blur(img)
        elif n == 1:
            return DataAugmentation.rotate(img)
        elif n == 2:
            return DataAugmentation.grayscale(img)
        elif n == 3:
            return DataAugmentation.colorrand(img)
        else:
            return DataAugmentation.random_crop(img)

class RefCOCO(Dataset):
    # This class is used to load the RefCOCO dataset, it is a subclass of Dataset.
    # It loads the dataset and to preprocess the images and the labels.
    # the requried parameters are:
    # annotations_file: the path to the pickle file containing the labels (RefCOCO folder provide a pickle file called refs(umd).p)
    # img_dir: the path to the folder containing the images (RefCOCO folder provide a subfolder called images)
    # model: the CLIP model
    # preprocess: the preprocessing function to be applied to the images
    # transform: the transformation to be applied to the images
    # target_transform: the transformation to be applied to the labels
    # device: the device to be used (cuda or cpu)

    def __init__(self, annotations_file, img_dir, model, preprocess=None, transform=None, target_transform=None, device = 'cuda', sample_size=42226, batch_size=None, split_type='train', augment_data=False):
        x = pd.read_pickle(annotations_file)
        self.img_texts = pd.DataFrame(x)
        self.img_texts = self.img_texts.loc[self.img_texts['split'] == split_type]
        #self.img_texts.to_csv('Test/Data/labels_'+split_type+'.csv', index=False)
        self.target_transform = target_transform
        self.device = device
        self.sample_size = sample_size
        self.img_dir = img_dir
        self.transform = transform
        self.preprocess = preprocess
        self.model = model
        self.batch_size = batch_size
        self.img, self.description = self.get_imgs_texts()
        info(split_type.upper()+": ENCODING"+(" & AUGMENTIG DATA..." if augment_data else "..."))
        self.enc_imgs, self.enc_txts = self.encode_data(augment_data)

    def __len__(self):
        return len(self.enc_imgs)

    def get_imgs_texts(self):
        images=[]
        texts=[]
        index=0
        for _, el in self.img_texts.iterrows():
            images.append(self.img_dir+"/"+el["file_name"][0:27]+".jpg")
            sentences=[]
            for sent in el["sentences"]:
                sentences.append(sent["raw"])
            texts.append(sentences)
            index+=1
            if index >= self.sample_size:
                break
        return images, texts
    
    

    
    def encode_data(self, augment_data):
        enc_imgs=[]
        for img in self.img:
            enc_imgs.append(self.preprocess(Image.open(img)))
            if augment_data:
                tmp = self.preprocess(Image.open(img))
                enc_imgs.append(DataAugmentation.random_augmentation(tmp))
                # enc_imgs.append(DataAugmentation.random_augmentation(tmp))
                # enc_imgs.append(DataAugmentation.random_augmentation(tmp))
                # enc_imgs.append(DataAugmentation.random_augmentation(tmp))

        enc_txts=[]
        for txt in self.description:
            enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
            if augment_data:
                enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
                # enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
                # enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
                # enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
        return enc_imgs, enc_txts
    
    
    def __getimg__(self, idx):
        image = self.img[idx]
        str_image = str(image)
        return image, str_image
    
    def __gettext__(self, idx):
        text = self.description[idx]
        return text
    
    def __getitem__(self, idx):
        # This function is used to get the item at the index idx
        # the required parameter is:
        # idx: the index of the item to be returned
        image = self.enc_imgs[idx]
        text = self.enc_txts[idx]
        
        return image, text
