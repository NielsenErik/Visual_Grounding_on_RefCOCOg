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

    def __init__(self, annotations_file, img_dir, model, preprocess=None, transform=None, target_transform=None, device = 'cuda', sample_size=5023, batch_size=None):
        x = pd.read_pickle(annotations_file)
        self.img_texts = pd.DataFrame(x)
        #This is to save the labels in a csv file, it is not necessary, it is helpful to check the labels
        #self.img_labels.to_csv('Test/Data/labels.csv',index=False)
        self.target_transform = target_transform
        self.device = device
        self.sample_size = sample_size
        self.img_dir = img_dir
        self.transform = transform
        self.preprocess = preprocess
        self.model = model
        self.clip_model, _ = clip.load("RN50", device=self.device)
        self.max_len_desc=0
        self.batch_size = batch_size
        self.img = self.get_img()
        self.description = self.get_texts()
        #self.tok_texts = self.tokenize_texts()
        #self.empty_tok_text=clip.tokenize("").squeeze(0)

    def __len__(self):
        return self.sample_size

    def get_img(self):
        # This function get the data inmages file names and the descriptions attached to them
        img_dir = Path(self.img_dir)
        image_names_tmp = [
            filename for filename in img_dir.glob('*')
            if filename.suffix in {'.png', '.jpg'}
        ]
        image_names =[image_names_tmp[i] for i in range(self.sample_size)]
        return image_names

    def get_texts(self):
        texts = []
        for x in self.img:
            ind = self.img_texts.index[self.img_texts["image_id"] == int(x.name.split("_")[2].split(".")[0])].tolist()
            desc = []
            for y in ind:
                desc_dict = self.img_texts.iat[y, 2]
                for z in desc_dict:
                    desc.append(z["raw"])
            self.max_len_desc=len(desc) if len(desc)>self.max_len_desc else self.max_len_desc
            texts.append(desc)
        #equal dimensions descriptions
        #for t in texts:
        #    if len(t) < self.max_len_desc:
        #        for i in range(self.max_len_desc-len(t)):
        #            t.append("")
        return texts 
    def set_empty_tok_text(self):
        texts_z = clip.tokenize("Empty").to(self.device)
        with torch.no_grad():
            texts_z = self.clip_model.encode_text(texts_z).float()
            texts_z /= texts_z.norm(dim=-1, keepdim=True)
        return texts_z
    
    def tokenize_texts(self, idx):
        if len(self.description[idx])==0:
            tok_texts = clip.tokenize("Empty").to(self.device)
        else:
            tok_texts = clip.tokenize(self.description[idx][0]).to(self.device)
        with torch.no_grad():
            texts_z = self.clip_model.encode_text(tok_texts).float()      
        #tok_texts /= tok_texts.norm(dim=-1, keepdim=True)
        return texts_z
    
    def __getimg__(self, idx):
        image = str(self.img[idx])
        return image
    
    def __gettext__(self, idx):
        text = self.description[idx]
        return text
   
    def __getitem__(self, idx):
        max_len_desc = 10
        # This function is used to get the item at the index idx
        # the required parameter is:
        # idx: the index of the item to be returned
        image = self.preprocess(Image.open(self.img[idx]))
        if self.transform:
            image = self.transform(image)
        text = self.tokenize_texts(idx).squeeze()
        # print(text.size())
        # if self.transform:
        #     image = self.transform(image)
        
        # if text.size(dim=0)>1:
        #      text = text[random.randint(0,text.size(dim=0)-1)]
        # elif len(text.size())<3:
        #     text = text.unsqueeze(0)
        # if len(text)>0:
        #     text = text[random.randint(0,len(text)-1)]
        # else:
        #     text = ""
        
        # if len(text)==0:
        #     text = ""
        
        
        # text = list(self.description[idx])
        # if len(text)==0:
        #     text = ""
        
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

    def __init__(self, annotations_file, img_dir, model, preprocess, split_type = 'test', transform=None, target_transform=None, device='cuda', sample_size=5023, batch_size=None):
        super().__init__(annotations_file, img_dir, model, preprocess, transform, target_transform, device, sample_size, batch_size)
        self.img_texts = self.img_texts.loc[self.img_texts['split'] == split_type]
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __getimg__(self, idx):
        return super().__getimg__(idx)
