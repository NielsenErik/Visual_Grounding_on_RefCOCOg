import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torchvision.transforms as T
from cocoLoad import RefCOCO_Split
from printCalls import info, error, debugging
from yolo_classes import get_yolo_classes
import clip
from PIL import Image



import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

#Here useful github page were there is an implementation
# of the baseline idea YOLO + Clip
# https://github.com/vijishmadhavan/Crop-CLIP/blob/master/Crop_CLIP.ipynb
# https://stackoverflow.com/questions/73593712/calculating-similarities-of-text-embeddings-using-clip
# https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb

#import YOLO model

get_device_first_call=True

def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def get_img_transform():
    transform = list()
    transform.append(T.ToPILImage())
    # resize each PIL image to 256 x 256
    transform.append(T.Resize((256, 256)))
    transform.append(T.CenterCrop((224,224)))   
    # convert Numpy to Pytorch Tensor    
    transform.append(T.ToTensor())  
    
    transform = T.Compose(transform)
    return transform

def get_data(batch_size, annotations_file, img_root, model, preprocess = None, device = get_device(), sample_size = 5023):
    #This function returns the training and test data loaders
    #The data loaders will be used by the training and test functions respectively
    #The data loaders will be used to load the data in batches of size batch_size
    #The data loaders will also apply the transformations to the data as specified in the transform function
    #batch_size: the batch size to be used
    #annotations_file: the path to the file containing the annotations
    #img_root: the path to the folder containing the images
    #transform: the transform function to be applied on the data
    #model: the model to be used for encoding the images and texts
    #preprocess: the preprocess function to be applied on the images
    #device: the device to be used for training
    #sample_size: the number of samples to be used for training and testing
    transform = get_img_transform()    
    training_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='train', transform=transform, device=device, sample_size=sample_size)
    test_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='test', transform=transform, device=device, sample_size=sample_size)
    num_training_samples = len(training_data)
    info("Number of training samples:" + str(num_training_samples))
    num_test_samples = len(test_data)
    info("Number of test samples:" + str(num_test_samples))
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def plot_beginning(data_loader, preprocess, n_samples = 12):
    # This function plots the first n_samples images in the data_loader
    # with their corresponding texts
    # data_loader: the data loader to be used
    # preprocess: the preprocess function to be applied on the images
    # n_samples: the number of samples to be plotted
    
    original_images = []
    images = []
    texts = []
    plt.figure(figsize=(16, 10))
    plt.suptitle("Sample images with corresponding texts")
    
    for batch_idx, (file_name, descs) in enumerate(data_loader):
        if len(descs)==0:
                continue        
        image = Image.open(file_name[0][0]).convert('RGB')
        if(len(images) >= n_samples):
            break
        tmp = ""
        for text in descs:
            tmp += text[0] + "\n"
        title_img = "File: " + str(file_name[0][0])+"\n"+tmp
        
        plt.subplot(3, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(title_img, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descs)
    plt.tight_layout()
    plt.savefig('Plots/beginning.png')

def zero_shot_plots(data_loader, res_probs, res_labels, n_samples = 16):

    pass
     
def eval_step(yolo_model, clip_model, clip_preprocess, data_loader, device=get_device(), yolo_threshold=0.5, clip_threshold=0.5):
    # This function evaluates the model on the data_loader
    # yolo_model: the yolo model to be used
    # clip_model: the clip model to be used
    # clip_preprocess: the preprocess function to be applied on the images
    # data_loader: the data loader to be used
    # device: the device to be used for training
    # yolo_threshold: the threshold to be used for yolo
    # clip_threshold: the threshold to be used for clip
    res_probs = []
    res_labels = []
    yolo_model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            debugging("In eval loop")
            input = inputs[0][0]
            PILimg = Image.open(input)
            outputs = yolo_model(input)
            result = outputs.pandas().xyxy[0]
            targets=[t[0] for t in targets]
            debugging("Targets: " + str(targets))
            if len(targets)==0:
                info("EMPTY TARGETS")
                continue
            for ind in result.index:
                debugging("Object: " + str(result["name"][ind]) + " - Confidence: " + str(result["confidence"][ind]))
                if result["confidence"][ind] > yolo_threshold:
                    #2 foreach oggetto: valutazione similarità oggetto_ritagliato-target con clip (https://huggingface.co/docs/transformers/model_doc/clip#:~:text=from%20PIL%20import%20Image%0A%3E%3E%3E%20import%20requests%0A%0A%3E%3E%3E,take%20the%20softmax%20to%20get%20the%20label%20probabilities)
                    PILcropped = PILimg.crop((int(result["xmin"][ind]), int(result["ymin"][ind]), int(result["xmax"][ind]), int(result["ymax"][ind])))
                    #clip_inputs = clip_processor(text=targets, images=PILcropped, return_tensors="pt", padding=True)
                    clip_inputs = clip_preprocess(PILcropped).unsqueeze(0).to(device)
            targets = clip.tokenize(["This is " + desc for desc in targets]).cuda()
            image_features = clip_model.encode_image(clip_inputs).float()
            text_features = clip_model.encode_text(targets).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = text_probs.cpu().topk(1)
            res_probs.append(top_probs)
            res_labels.append(top_labels)
    return res_probs, res_labels
           
def main(num_samples = 50):
    # This is the main function that will be called to train the model
    info("Starting baseline")
    batch_size = 1
    device = get_device()
    learning_rate = 0.001
    weight_decay = 0.000001
    momentum = 0.9
    epochs = 50,
    num_classes = 65,
    annotations_file = 'refcocog/annotations/refs(umd).p'
    root_imgs = 'refcocog/images'
    
    yolo_classes = get_yolo_classes()
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    clip_model, clip_preprocess = clip.load("RN50", device=device)
    
    train_loader, test_loader = get_data(batch_size, annotations_file, root_imgs, clip_model, clip_preprocess, device, sample_size=num_samples)
    #plot_beginning(test_loader, clip_preprocess)
    info("Starting evaluation")
    res_probs, res_labes = eval_step(yolo_model, clip_model, clip_preprocess, test_loader, device)
    zero_shot_plots(test_loader, res_probs, res_labes)
main()