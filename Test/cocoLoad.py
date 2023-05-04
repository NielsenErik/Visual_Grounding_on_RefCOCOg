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
        self.encoded_img = self.encode_img()
        self.description = self.get_texts()

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
        # desc = []
        # texts = []
        # for j in range(self.sample_size):
        #     for i in range(len(self.img_texts.iloc[j, 2])):
        #         desc.append(self.img_texts.iloc[j, 2][i]["raw"]) #TODO: FIX THIS size to match img sizes
        #     texts.append(desc)
        return image_names

    def get_texts(self):
        desc = []
        texts = []
        for j in range(self.sample_size):
            for i in range(len(self.img_texts.iloc[j, 2])):
                desc.append(self.img_texts.iloc[j, 2][i]["raw"]) #TODO: FIX THIS size to match img sizes
            texts.append(desc)
        return texts

    def encode_img(self):
        # This function encode the images data and the text data
        # the required parameters are:
        # images_fp: the list of the images file names
        # texts: the list of the descriptions attached to the images
        debugging("In encode_data")
        image_names = self.get_img()
        open_img = [Image.open(image) for image in image_names]
        images = [self.preprocess(image) for image in open_img]
        images = torch.tensor(np.stack(images)).to(self.device)
        #with torch.no_grad():
            #images_z = self.model.encode_image(images).float()
            #texts_z = self.model.encode_text(text_tokens).float()
        return images

    def encode_texts(self, desc_fp):#TODO: FIX size of target tensor
        debugging("In encode_data: tokenize descriptions")
        text_tokens = clip.tokenize(desc_fp).to(self.device)
        #text_tokens = torch.tensor(text_tokens).to(self.device)
        #with torch.no_grad():
        #    texts_z = self.model.encode_text(text_tokens).float()
        return text_tokens       
        
   
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
        debugging("In getitem")
        image = self.encoded_img[idx]
        # if self.transform:
        #      image = self.transform(image_)
        img_desc = self.description[idx]
        texts = self.encode_texts(img_desc)
        debugging("In getitem: return")
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
    # device: the device to be used, it can be 'cuda' or 'cpu'
    # sample_size: the size of the dataset to be loaded

    def __init__(self, annotations_file, img_dir, model, preprocess, split_type = 'test', transform=None, target_transform=None, device='cuda', sample_size=5023):
        super().__init__(annotations_file, img_dir, model, preprocess, transform, target_transform, device, sample_size)
        self.img_texts = self.img_texts.loc[self.img_texts['split'] == split_type]
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)
