import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from printCalls import debugging, info
from PIL import Image
import clip
import random

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

    def __init__(self, annotations_file, img_dir, model, preprocess=None, transform=None, target_transform=None, device = 'cuda', sample_size=5023, batch_size=None, split_type='train'):
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
        self.enc_txts = self.encode_texts()
        self.enc_imgs = self.encode_images()

    def __len__(self):
        return len(self.img)

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
    
    
    def encode_texts(self):
        enc_txts=[]
        for txt in self.description:
            enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
        return enc_txts
    
    def encode_images(self):
        enc_imgs=[]
        for img in self.img:
            enc_imgs.append(self.preprocess(Image.open(img)))
        return enc_imgs
    
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
        #image = self.preprocess(Image.open(self.img[idx]))
        image = self.enc_imgs[idx]
        text = self.enc_txts[idx]        
        
        return image, text

class RefCOCO_Split(RefCOCO):
    # This class is used to load the RefCOCO dataset, it is a subclass of Dataset.
    # It loads the dataset and to preprocess the images and the labels and then
    # it splits the dataset in training and test set depending on the split_type parameter.
    # the requried parameters are:
    # annotations_file: the path to the pickle file containing the labels (RefCOCO folder provide a pickle file called refs(umd).p)
    # img_dir: the path to the folder containing the images (RefCOCO folder provide a subfolder called images)
    # split_type: the type of split to be used, it can be 'train' or 'test'
    # transform: the transformation to be applied to the images
    # target_transform: the transformation to be applied to the labels
    # device: the device to be used, it can be 'cuda' or 'cpu'
    # sample_size: the size of the dataset to be loaded

    def __init__(self, annotations_file, img_dir, model, preprocess, split_type = 'train', transform=None, target_transform=None, device='cuda', sample_size=5023, batch_size=None):
        super().__init__(annotations_file, img_dir, model, preprocess, transform, target_transform, device, sample_size, batch_size, split_type)
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __getimg__(self, idx):
        return super().__getimg__(idx)
