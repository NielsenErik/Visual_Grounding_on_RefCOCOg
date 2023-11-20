import torch
import torchvision
import torchvision.transforms as T
from cocoLoad import RefCOCO
from printCalls import info, error, debugging, warning
from yolo_classes import get_yolo_classes
import clip
from PIL import Image
import cv2
import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def get_all_texts(annotations_file):
    x = pd.read_pickle(annotations_file)
    img_texts = pd.DataFrame(x)
    all_texts = []
    for x in img_texts.iloc():
        if x['split']=='test' and len(x['sentences'][0]['raw'])>0:
            all_texts.append(x['sentences'][0]['raw'])
    #info("Number of texts: " + str(len(all_texts)))
    return all_texts

def putTextBg(img, text, org, font, size, fg_color, thickness, linetype, bg_color):
    text_size, _ = cv2.getTextSize(text, font, size, thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, (org[0]-2, org[1]-text_h-2), (org[0] + text_w + 2, org[1] + 5), bg_color, -1)
    img = cv2.putText (img, text, org, font, size, fg_color, thickness, linetype)
    return img

get_device_first_call=True
def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def get_cost_function(isImg=True):
    return torch.nn.CrossEntropyLoss()

def get_data(batch_size, annotations_file, img_root, model, test_batch_size = 16, preprocess = None, device = get_device(), sample_size_val = 2573):
    sample_size_val = sample_size_val if sample_size_val <= 2573 else 2573
    eval_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='val', device=device, sample_size=sample_size_val, batch_size=test_batch_size)
    num_eval_samples = len(eval_data)
    info("Number of eval samples:" + str(num_eval_samples))
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=test_batch_size, shuffle=False)
    return eval_loader

def test_step(model, test_loader, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    model.eval() 
    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for (images, texts) in test_loader:
            #images, texts = batch
            images = images.to(device)
            texts = texts.squeeze(1).to(device)
            logits_per_image, logits_per_texts = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            img_loss = cost_function(logits_per_image, ground_truth)
            desc_loss = cost_function(logits_per_texts, ground_truth)
            loss = (img_loss + desc_loss)/2
            #debugging(str(loss.item()))
            cumulative_loss += loss.item() 
            samples += images.shape[0]  
            _, predicted = logits_per_image.max(dim=1)    
            cumulative_accuracy += predicted.eq(ground_truth).sum().item()
        
    return cumulative_loss / samples, cumulative_accuracy / samples

def main():
    batch_size = 16 #must be 16 due to lenght of clip_targets
    test_batch_size = 8
    device = 'cuda:0'
    cost_function = get_cost_function()
    learning_rate = 0.001
    weight_decay = 0.000001
    momentum = 0.9
    epochs = 30
    e = math.exp(1)
    alpha = 1
    
    visualization_name='RefCOCOg'

    annotations_file = 'refcocog/annotations/refs(umd).p'
    root_imgs = 'refcocog/images'
    clip_model, clip_processor = clip.load("RN50", device=device)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
    eval_loader = get_data(batch_size, annotations_file, root_imgs, clip_model, clip_processor, device, sample_size_val=2573)
    yolo_classes = get_yolo_classes()