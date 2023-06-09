import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import clip
import math

#Custom modules
from cocoLoad import RefCOCO #Importing REfCOCO class from cocoLoad.py
from printCalls import error, warning, debugging, info 
from customClip import CustomClip
from model_utilis import save_model, load_model, TensorBoard
import final_program
from evaluation_metrics import recall, intersection_over_union, semantic_similarity


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
#RUNNING are the following:
# CrossEntropyLoss()
# HingeEmbeddingLoss()
# SmoothL1Loss()

def get_optimizer(net, lr, wd):
    return torch.optim.Adadelta(net.parameters(), lr=lr, weight_decay = wd)

# NO Adam
# OK ASGD run 10
# OK SGD run 11 Best
# NO RMSprop run 12
# No Nadam run 13
# No LBFGS run 14
# OK Adadelta run 15 seem GOOOODDD
# OK SGD Big Data run 16
# OK Adadelta Big Data run 17

def update_parameters(learning_rate, weight_decay, alpha):
    learning_rate = learning_rate * alpha
    weight_decay = weight_decay * alpha
    alpha = alpha/(alpha+0.001)
    return learning_rate, weight_decay, alpha


def get_data(batch_size, annotations_file, img_root, model, test_batch_size = 16, preprocess = None, device = get_device(), sample_size_train = 42226, sample_size_test = 5023, sample_size_val = 2573, augment_data_train=True):
    sample_size_train = sample_size_train if sample_size_train <= 42226 else 42226
    sample_size_test = sample_size_test if sample_size_test <= 5023 else 5023
    sample_size_val = sample_size_val if sample_size_val <= 2573 else 2573
    training_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='train', device=device, sample_size=sample_size_train, batch_size=batch_size, augment_data=augment_data_train)
    test_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='test', device=device, sample_size=sample_size_test, batch_size=test_batch_size)
    eval_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='val', device=device, sample_size=sample_size_val, batch_size=test_batch_size)

    num_training_samples = len(training_data)
    info("Number of training samples:" + str(num_training_samples))
    num_test_samples = len(test_data)
    info("Number of test samples:" + str(num_test_samples))
    num_eval_samples = len(eval_data)
    info("Number of eval samples:" + str(num_eval_samples))
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, eval_loader


def training_step(model, train_dataloader,  optimizer, cost_function=get_cost_function(), device=get_device()):
    cumulative_accuracy = 0.0
    cumulative_loss = 0.0
    samples = 0.0
    clip.model.convert_weights(model)
    model.train()
    for (images, texts) in train_dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        texts = texts.squeeze(1).to(device)
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        logits_per_image, logits_per_texts = model(images, texts)
        img_loss = cost_function(logits_per_image, ground_truth)
        desc_loss = cost_function(logits_per_texts, ground_truth)
        loss = (img_loss + desc_loss)/2
        loss.backward()
        optimizer.step()        
        cumulative_loss += loss.item()    
        samples += images.shape[0]  
        _, predicted = logits_per_image.max(dim=1)    
        cumulative_accuracy += predicted.eq(ground_truth).sum().item()
        clip.model.convert_weights(model)
    return cumulative_loss / samples, cumulative_accuracy / samples


def test_step(model, test_loader, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    model.eval() 
    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for (images, texts) in test_loader:
            images = images.to(device)
            texts = texts.squeeze(1).to(device)
            logits_per_image, logits_per_texts = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            img_loss = cost_function(logits_per_image, ground_truth)
            desc_loss = cost_function(logits_per_texts, ground_truth)
            loss = (img_loss + desc_loss)/2
            cumulative_loss += loss.item() 
            samples += images.shape[0]
            _, predicted = logits_per_image.max(dim=1)
            cumulative_accuracy += predicted.eq(ground_truth).sum().item()
        
    return cumulative_loss / samples, cumulative_accuracy / samples


def eval_step(model, eval_loader, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    comulative_recall = 0.0
    model.eval() 
    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the set
        for (images, texts) in eval_loader:
            images = images.to(device)
            texts = texts.squeeze(1).to(device)
            logits_per_image, logits_per_texts = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            img_loss = cost_function(logits_per_image, ground_truth)
            desc_loss = cost_function(logits_per_texts, ground_truth)
            loss = (img_loss + desc_loss)/2
            cumulative_loss += loss.item() 
            samples += images.shape[0]  
            n_labels = logits_per_texts.shape[1]
            _, predicted = logits_per_image.max(dim=1)
            cumulative_accuracy += predicted.eq(ground_truth).sum().item()
            comulative_recall += recall(predicted, ground_truth, n_labels, device)

    return cumulative_loss / samples, cumulative_accuracy / samples, comulative_recall / samples


def main():

    #DATASET PARAMS
    sample_size_train=18000
    sample_size_test=5120
    sample_size_val=256
    augment_data_train=True

    #TRAINING PARAMS
    batch_size = 32 #must be 16 due to lenght of clip_targets
    test_batch_size = 64
    epochs = 30
    
    #OPTIMIZER & LOSS PARAMS
    cost_function = get_cost_function()
    learning_rate = 0.0015
    weight_decay = 0.000001
    alpha = 1


    annotations_file = 'refcocog/annotations/refs(umd).p'
    root_imgs = 'refcocog/images'
    clip_model = CustomClip(device=get_device(), batch_size=batch_size, norm=False, bias=True)
    _ , clip_processor = clip_model.__get_model__()
    optimizer = get_optimizer(clip_model, learning_rate, weight_decay)

    train_loader, test_loader, val_loader = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs, model=clip_model, test_batch_size = test_batch_size, preprocess=clip_processor, sample_size_train=sample_size_train, sample_size_test=sample_size_test, sample_size_val=sample_size_val, augment_data_train=augment_data_train)

    
    tb = TensorBoard("run")
    
    info("BEFORE TRAINING...")

    loss, accuracy = test_step(clip_model, train_loader, cost_function)
    info("Train - LOSS: {:.4} ACCURACY: {:2.1%}".format(loss, accuracy))
    tb.log_values(epochs+1, loss, accuracy, "Train")
    loss, accuracy = test_step(clip_model, val_loader, cost_function)
    info("Validation - LOSS: {:.4} ACCURACY: {:2.1%}".format(loss, accuracy))
    tb.log_values(epochs+1, loss, accuracy, "Validation")
    loss, accuracy = test_step(clip_model, test_loader, cost_function)
    info("Test - LOSS: {:.4} ACCURACY: {:2.1%}".format(loss, accuracy))
    tb.log_values(epochs+1, loss, accuracy, "Test")
    optimizer = get_optimizer(clip_model, learning_rate, weight_decay)    

    info("INIT TRAINING...")
    for ep in range(1, epochs+1):
        info("EPOCH "+str(ep)+":")
        loss, accuracy = training_step(clip_model, train_loader, optimizer, cost_function)
        if ep % 5 == 0:
            save_model(clip_model, ep, optimizer, loss, "Personal_Model")
        info("Train - LOSS: {:.4} ACCURACY: {:2.1%}% ".format(loss, accuracy))
        tb.log_values(ep, loss, accuracy, "Train")
        loss, accuracy = test_step(clip_model, val_loader, cost_function)
        info("Validation - LOSS: {:.4} ACCURACY: {:2.1%}%".format(loss, accuracy))
        tb.log_values(ep, loss, accuracy, "Validation") 
        learning_rate, weight_decay, alpha = update_parameters(learning_rate, weight_decay, alpha)
        optimizer = get_optimizer(clip_model, learning_rate, weight_decay)

    info("AFTER TRAINING...")
    loss, accuracy, recall = eval_step(clip_model, train_loader, cost_function)
    info("Train - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Train")
    loss, accuracy, recall = eval_step(clip_model, val_loader, cost_function)
    info("Validation - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Validation")
    loss, accuracy, recall = eval_step(clip_model, test_loader, cost_function)
    info("Test - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Test")
    tb.close()

    save_model(clip_model, epochs, optimizer, loss, "Personal_Model")

##########################################################################################
main()
