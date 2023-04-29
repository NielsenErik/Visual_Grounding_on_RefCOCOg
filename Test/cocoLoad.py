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
    # transform: the transformation to be applied to the images
    # target_transform: the transformation to be applied to the labels

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        x = pd.read_pickle(annotations_file)
        self.img_labels = pd.DataFrame(x)
        #self.img_labels.to_csv('Test/Data/labels.csv',index=False)
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        file_names = self.img_labels['file_name']#id string must be converted in 12 int digits 0000..xyzasd.jpg
        remove_id = self.img_labels['ann_id']
        file_name = file_names.iloc[idx].replace('_'+str(remove_id.iloc[idx])+'.jpg', '.jpg')
        image_name = os.path.join(self.img_dir, file_name)
        image_name = read_image(image_name)
        label = self.img_labels.iloc[idx, 2][0]["raw"] #this are the lables shown as tuples, this must be fixed
        label = clip.tokenize(label)
        if self.transform:
            image = self.transform(image_name)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
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

    def __init__(self, annotations_file, img_dir, split_type = 'train', transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, transform, target_transform)
        self.img_labels = self.img_labels.loc[self.img_labels['split'] == split_type]
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
