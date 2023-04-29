import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.nn.functional as F
import torchvision.transforms as T
import os
from torchvision.io import read_image
from cocoLoad import RefCOCO, RefCOCO_Split #Importing REfCOCO class from cocoLoad.py
from clip import clip


# import yolo baseline architecture
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, _verbose=False)

# get the cost function
def get_cost_function():
    return torch.nn.CrossEntropyLoss()

# get the optimization algorithm
def get_optimizer(net, lr, wd, momentum):
    return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

# check if Cuda is available to be used by the device, otherwise uses cpu


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("The current device is", device)
    return device

def get_img_transform():
    transform = list()
    transform.append(T.ToPILImage())
    # resize each PIL image to 256 x 256
    transform.append(T.Resize((256, 256)))
    transform.append(T.CenterCrop((224,224)))   
    # convert Numpy to Pytorch Tensor    
    transform.append(T.ToTensor())  
    transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))  
    transform = T.Compose(transform)
    return transform


def get_data(batch_size, annotations_file, img_root):
    transform = get_img_transform()
    #refCOCO_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, transform=transform)
    
    
    # In refCOCO there is already the plits inside the labels,
    #so we do't have to do the random split
    # training_samples = int(num_samples * 0.8+1)
    # test_samples = num_samples - training_samples
    # training_data, test_data = torch.utils.data.random_split(
    #     refCOCO_data, [training_samples, test_samples])
    training_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, split_type='train', transform=transform)
    test_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, split_type='test', transform=transform)

    num_training_samples = len(training_data)
    print("Number of training samples:", num_training_samples)
    num_test_samples = len(test_data)
    print("Number of test samples:", num_test_samples)

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def training_step(net, data_loader, optimizer, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    net.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        print("OK")
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = cost_function(outputs, targets)
        loss.backward()
        optimizer.step()
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        cumulative_accuracy += predicted.eq(inputs).sum().item()
    return cumulative_loss/samples, cumulative_accuracy/samples*100


def test_step(net, data_loader, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            print(inputs.size(), targets.size())
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            print(outputs[0].size())
            print(outputs[1][0].size())
            print(outputs[1][1].size())
            print(outputs[1][2].size())
            loss = cost_function(outputs[0], targets)
            loss.backward()
            samples += inputs.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            cumulative_accuracy += predicted.eq(targets).sum().item()
    return cumulative_loss/samples, cumulative_accuracy/samples*100

# model, preprocess = clip.load('RN50', device=get_device())
# print(clip.available_models())

# preliminar step


batch_size = 128
device = 'cuda:0'
learning_rate = 0.001
weight_decay = 0.000001
epochs = 50,
num_classes = 65,
annotations_file = 'refcocog/annotations/refs(umd).p'
root_imgs = 'refcocog/images'
train_loader, test_loader = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs)

print("Before training")
train_loss, train_accuracy = test_step(model, train_loader, get_cost_function())
test_loss, test_accuracy = test_step(model, test_loader, get_cost_function())
print('\tTraining loss {:.5f}, Training accuracy {:.2f}'.format(
    train_loss, train_accuracy))
print('\tTest loss {:.5f}, Test accuracy {:.2f}'.format(
    test_loss, test_accuracy))
print('-----------------------------------------------------')
