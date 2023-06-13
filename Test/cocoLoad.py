import pandas as pd
import math
from torch.utils.data import Dataset
from pathlib import Path
from printCalls import info
from PIL import Image
import clip
import random
from torchvision import transforms

class DataAugmentation():
    # This class is used to perform tranformation in order to have augmented data   
    
    def blur(img):
        # Gaussian Blur the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),            
        ])
        img = transform(img)
        return img
    
    def rotate(img):
        # Rotate the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
    def grayscale(img):
        # Convert the image to grayscale
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
    def colorrand(img):
        # Randomly change the color of the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.01*random.randrange(1,50), contrast=0.01*random.randrange(1,50), saturation=0.01*random.randrange(1,50), hue=0.01*random.randrange(1,50)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
    def random_crop(img):
        # Randomly crop the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        return img
        
    def random_augmentation(img):
        # Randomly choose a transformation to be applied to the image
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
    # sample_size: the number of samples to be loaded in the dataset
    # batch_size: the batch size when __get_item__ is called
    # split_type: the split to be loaded (train, val, test)
    # augment_data: if True, the data is augmented with random transformations

    def __init__(self, annotations_file, img_dir, preprocess, device, sample_size, split_type, augment_data=False, with_boxes=False):
        x = pd.read_pickle(annotations_file)
        self.img_texts = pd.DataFrame(x)
        self.img_texts = self.img_texts.loc[self.img_texts['split'] == split_type]
        self.device = device
        self.sample_size = sample_size
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.img, self.description = self.get_imgs_texts(with_boxes)
        info(split_type.upper()+": ENCODING"+(" & AUGMENTIG DATA..." if augment_data else "..."))
        self.enc_imgs, self.enc_txts = self.encode_data(augment_data, with_boxes)

    def __len__(self):
        return len(self.enc_imgs)

    def get_imgs_texts(self, with_boxes):
        # This function is used to get the images and the labels (descriptions) from the dataset
        # It returns a list of images and a list of list of sentences ordered (idx image = idx text)
        images=[]
        texts=[]
        self.boxes=[]
        index=0
        for _, el in self.img_texts.iterrows():
            images.append(self.img_dir+"/"+el["file_name"][0:27]+".jpg")
            if with_boxes:
                if math.isnan(el["xmin"]) or math.isnan(el["ymin"]) or math.isnan(el["xmax"]) or math.isnan(el["ymax"]):
                    self.boxes.append({"valid": False})
                else:
                    self.boxes.append({"valid": True, "xmin":int(el["xmin"]), "ymin":int(el["ymin"]), "xmax":int(el["xmax"]), "ymax":int(el["ymax"])})
            sentences=[]
            for sent in el["sentences"]:
                sentences.append(sent["raw"])
            texts.append(sentences)
            index+=1
            if index >= self.sample_size:
                break
        return images, texts
    
    def encode_data(self, augment_data, with_boxes):
        # This function is used to encode data. It executes:
        # - Tokenization for text
        # - Preprocessing for images
        # - Randomly data augmentation if requested (1 original img = 1 noisy img)
        # the required parameter is:
        # augment_data: boolean that enable data augmentation
        enc_imgs=[]
        for i, img in enumerate(self.img):
            tmp = Image.open(img)
            if with_boxes and self.boxes[i]["valid"]:
                tmp = tmp.crop((self.boxes[i]["xmin"], self.boxes[i]["ymin"], self.boxes[i]["xmax"], self.boxes[i]["ymax"]))
            enc_imgs.append(self.preprocess(tmp))
            if augment_data:
                enc_imgs.append(DataAugmentation.random_augmentation(self.preprocess(tmp)))

        enc_txts=[]
        for txt in self.description:
            enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))
            if augment_data:
                enc_txts.append(clip.tokenize(txt[random.randint(0, len(txt)-1)]).to(self.device))

        return enc_imgs, enc_txts
    
    
    def __getimg__(self, idx):
        # This function is used to get image path and image name at index idx
        # Used only for evaluation purposes
        # the required parameter is:
        # idx: the index of the item to be returned
        image = self.img[idx]
        str_image = str(image)
        return image, str_image
    
    def __gettext__(self, idx):
        # This function is used to get list of descriptions at index idx
        # Used only for evaluation purposes
        # the required parameter is:
        # idx: the index of the item to be returned
        text = self.description[idx]
        return text
    
    def __getitem__(self, idx):
        # This function is used to get the item at the index idx
        # the required parameter is:
        # idx: the index of the item to be returned
        image = self.enc_imgs[idx]
        text = self.enc_txts[idx]
        
        return image, text
