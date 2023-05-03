import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from printCalls import debugging, step
from PIL import Image
import clip

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
    
    def __init__(self, annotations_file, img_dir, model, preprocess, transform=None, target_transform=None, device = 'cuda', sample_size=5023):
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
        self.encoded_img, self.encoded_texts = self.encode_data()

    def __len__(self):
        return len(self.img_texts)
    
    def get_data(self):
        # This function get the data inmages file names and the descriptions attached to them
        img_dir = Path(self.img_dir)
        image_names_tmp = [
            filename for filename in img_dir.glob('*')
            if filename.suffix in {'.png', '.jpg'}
        ] 
        image_names =[image_names_tmp[i] for i in range(self.sample_size)]   
        #desc = []
        texts = []
        for j in range(self.sample_size):
            for i in range(len(self.img_texts.iloc[j, 2])):
                texts.append(self.img_texts.iloc[j, 2][i]["raw"]) #this are the lables shown as tuples, this must be fixed
  
        return image_names, texts
        
    def encode_data(self):
        # This function encode the images data and the text data
        # the required parameters are:
        # images_fp: the list of the images file names
        # texts: the list of the descriptions attached to the images
        debugging("In encode_data")
        image_names, desc = self.get_data()
        open_img = [Image.open(image) for image in image_names]
        images = [self.preprocess(image) for image in open_img] 
        images = torch.tensor(np.stack(images)).to(self.device)
        debugging("In encode_data: tokenize descriptions")
        text_tokens = clip.tokenize(desc).to(self.device)
        debugging("In encode_data: text tokens")
        with torch.no_grad():
            images_z = self.model.encode_image(images).float()
            texts_z = self.model.encode_text(text_tokens).float() 
        return images_z, texts_z        
        
   
    def __getitem__(self, idx):
        
        # This function is used to get the item at the index idx
        # the required parameter is:
        # idx: the index of the item to be returned

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
        image_ = self.encoded_img[idx]
        if self.transform:
            image = self.transform(image_)
        texts = self.encoded_texts[idx]
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

    def __init__(self, annotations_file, img_dir, model, preprocess, split_type = 'test', transform=None, target_transform=None, device='cuda', sample_size=5023):
        super().__init__(annotations_file, img_dir, model, preprocess, transform, target_transform, device, sample_size)
        self.img_texts = self.img_texts.loc[self.img_texts['split'] == split_type]
    def __len__(self):
        return len(self.img_texts)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
