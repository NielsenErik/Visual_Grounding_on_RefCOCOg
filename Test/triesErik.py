import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms as T
from cocoLoad import RefCOCO, RefCOCO_Split #Importing REfCOCO class from cocoLoad.py
from transformers import CLIPProcessor, CLIPModel
import clip
from printCalls import error, warning, debugging, info 

get_device_first_call=True
def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def get_cost_function():
    return torch.nn.CrossEntropyLoss()

def get_optimizer(net, lr, wd, momentum):
    return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

def get_img_transform():
    transform = list()
    transform.append(T.ToPILImage())
    # resize each PIL image to 256 x 256
    transform.append(T.Resize((256, 256)))
   # transform.append(T.CenterCrop((224,224)))   
    # convert Numpy to Pytorch Tensor    
    transform.append(T.ToTensor())  
    
    transform = T.Compose(transform)
    return transform

def get_data(batch_size, annotations_file, img_root, model, preprocess = None, device = get_device(), sample_size = 5023):
    transform = get_img_transform()    
    training_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='train', transform=transform, device=device, sample_size=sample_size)
    test_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='test', transform=transform, device=device, sample_size=sample_size)
    num_training_samples = len(training_data)
    info("Number of training samples:" + str(num_training_samples))
    num_test_samples = len(test_data)
    info("Number of test samples:" + str(num_test_samples))
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_data

def training_step(model, train_dataloader,  optimizer, loss_img=get_cost_function(), device=get_device()):
    #https://github.com/openai/CLIP/issues/83#:~:text=for%20epoch%20in%20range,convert_weights(model)
    cumulative_accuracy = 0.0
    cumulative_loss = 0.0
    samples = 0.0
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        images, texts = batch         
        images = images.to(device)
        texts = texts.to(device)
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        #debugging(str(logits_per_image))
        loss = loss_img(logits_per_image, ground_truth)
        loss.backward()
        #debugging(str(loss.item()))
        cumulative_loss += loss.item() 
        samples += images.shape[0]  
        _, predicted = logits_per_image.max(dim=1)    
        cumulative_accuracy += predicted.eq(ground_truth).sum().item()

        if device == "cpu":
            optimizer.step()
        else : 
            for p in model.parameters(): 
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 
            optimizer.step()
            #clip.model.convert_weights(clip_model)

    debugging(str(cumulative_loss))    
    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def test_step(net, test_loader, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    net.eval() 
    debugging("Testing")
  # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, batch in enumerate(test_loader) :
            # load data into GPU
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            # forward pass
            logits_per_image, logits_per_text = net(inputs, targets)
            # loss computation
            ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=device)
             #debugging(str(logits_per_image))
            loss = cost_function(logits_per_image, ground_truth)

            # fetch prediction and loss value
            samples += inputs.shape[0]
            cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = logits_per_image.max(1)

            # compute accuracy
            cumulative_accuracy += predicted.eq(ground_truth).sum().item()
        
    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def eval_step(clip_model, clip_processor, test_data):   
    
    pass

def main():
    batch_size = 8
    device = 'cuda:0'
    cost_function = get_cost_function()
    learning_rate = 0.001
    weight_decay = 0.000001
    momentum = 0.9
    epochs = 50
    annotations_file = 'refcocog/annotations/refs(umd).p'
    root_imgs = 'refcocog/images'

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    clip_model, clip_processor = clip.load('RN50', device=device, jit=False)

    optimizer = get_optimizer(clip_model, learning_rate, weight_decay, momentum)

    train_loader, test_loader, test_data = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs, model=clip_model, preprocess=clip_processor, sample_size=512)
    #eval_step(yolo_model, clip_model, clip_processor, test_data)
    for ep in range(epochs):
        info("EPOCH "+str(ep)+":")
        loss, accuracy = training_step(clip_model, train_loader, optimizer)
        info("LOSS: "+str(loss)+" ACCURACY: "+str(accuracy)+"%")
        clip.model.convert_weights(clip_model)
    info("TESTING:")
    loss, accuracy =test_step(clip_model, test_loader, cost_function)
    info("LOSS: "+str(loss)+" ACCURACY: "+str(accuracy)+"%")  

##########################################################################################
main()