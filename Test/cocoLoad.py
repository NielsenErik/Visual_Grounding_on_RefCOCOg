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
    def __init__(self, annotations_file, img_dir, split_type = 'train', transform=None, target_transform=None):
        super().__init__(annotations_file, img_dir, transform, target_transform)
        self.img_labels = self.img_labels.loc[self.img_labels['split'] == split_type]
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    

#refer = REFER(data_root, dataset='refcocog', splitBy='umd') # dataset='refcoco', splitBy='unc'    
    
# data = RefCOCO(annotations_file = 'refcocog/annotations/refs(umd).p', img_dir = 'refcocog/images')
# def get_data(batch_size, annotations_file, img_root):
    # transform = list()
    # # resize each PIL image to 256 x 256
    # transform.append(T.Resize((256, 256)))
    # # convert Numpy to Pytorch Tensor
    # transform.append(T.ToPILImage())
    # transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225]))
    
    
    # transform = T.Compose(transform)
    # refCOCO_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, transform=transform)
    # num_samples = len(refCOCO_data)
#     training_samples = int(num_samples * 0.8+1)
#     test_samples = num_samples - training_samples
#     training_data, test_data = torch.utils.data.random_split(
#         refCOCO_data, [training_samples, test_samples])
#     train_loader = torch.utils.data.DataLoader(
#         training_data, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(
#         test_data, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader

# loader,_ = get_data(1, 'refcocog/annotations/refs(umd).p', 'refcocog/images')
# i=0
# for x in loader:
#     i = i+1
#     print(i)