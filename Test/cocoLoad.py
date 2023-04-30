import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from clip import clip

from PIL import Image

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

    def __init__(self, annotations_file, img_dir, model, preprocess, transform=None, target_transform=None):
        x = pd.read_pickle(annotations_file)
        self.img_texts = pd.DataFrame(x)
        #This is to save the labels in a csv file, it is not necessary, it is helpful to check the labels
        #self.img_labels.to_csv('Test/Data/labels.csv',index=False) 
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.transform = transform
        self.preprocess = preprocess
        self.model = model

    def __len__(self):
        return len(self.img_texts)
    
    def get_data(self):
        file_names = self.img_texts['file_name']#id string must be converted in 12 int digits 0000..xyzasd.jpg
        remove_id = self.img_texts['ann_id']
        file_names = file_names.iloc[:].replace('_'+str(remove_id.iloc[:])+'.jpg', '.jpg')
        image_names = [os.path.join(self.img_dir, name) for name in file_names]
        desc = []
        texts = []
        for j in range(len(self.img_texts)):
            for i in range(len(self.img_texts.iloc[j, 2])):
                desc.append(self.img_texts.iloc[j, 2][i]["raw"]) #this are the lables shown as tuples, this must be fixed
            texts.append(desc)
        return image_names, texts
        
    def encode_data(self, images_fp: list[str], texts: list[str]):
        images = [self.preprocess(Image.open(image)) for image in images_fp]
        images = torch.tensor(np.stack(images)).cuda()
        text_tokens = clip.tokenize(texts).cuda()
        with torch.no_grad():
            images_z = self.model.encode_image(images).float()
            texts_z = self.model.encode_text(text_tokens).float()  
        return images_z, texts_z        
        
   
    def __getitem__(self, idx):

        # file_names = self.img_texts['file_name']#id string must be converted in 12 int digits 0000..xyzasd.jpg
        # remove_id = self.img_texts['ann_id']
        # file_name = file_names.iloc[idx].replace('_'+str(remove_id.iloc[idx])+'.jpg', '.jpg')
        # image_name = os.path.join(self.img_dir, file_name)
        # #image = read_image(image_name)
        # desc = []
        # texts = []
        # for i in range(len(self.img_texts.iloc[idx, 2])):
        #     desc.append(self.img_texts.iloc[idx, 2][i]["raw"]) #this are the lables shown as tuples, this must be fixed
        # texts.append(desc)
        # print(texts)
        # desc = clip.tokenize(desc)
        # if self.transform:
        #     image = self.transform(image_name)
        # if self.target_transform:
        #     desc = self.target_transform(desc)
        image_names, desc = self.get_data()
        images, texts_ = self.encode_data(image_names, desc)
        image = images[idx]
        texts = texts_[idx]
        return image, texts
    
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

    def __init__(self, annotations_file, img_dir, model, preprocess,  split_type = 'train', transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, model, preprocess, transform, target_transform)
        self.img_texts = self.img_texts.loc[self.img_texts['split'] == split_type]
    
    def __len__(self):
        return len(self.img_texts)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
