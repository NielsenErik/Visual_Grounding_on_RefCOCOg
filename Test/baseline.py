import torch
import torchvision
import torchvision.transforms as T
from cocoLoad import RefCOCO_Split
from printCalls import info, error, debugging, warning
from yolo_classes import get_yolo_classes
import clip
from PIL import Image
import cv2
import os
import skimage
import matplotlib.pyplot as plt
import numpy as np

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
def get_yolo_sentence(device):
    YOLO_CLASSES = get_yolo_classes()
    yolo_sentence = []
    for el in YOLO_CLASSES:
        yolo_sentence.append("This image contains a "+ str(el))
    clip_target = clip.tokenize(yolo_sentence).to(device)
    return clip_target, yolo_sentence

def putTextBg(img, text, org, font, size, fg_color, thickness, linetype, bg_color):
    text_size, _ = cv2.getTextSize(text, font, size, thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, (org[0]-2, org[1]-text_h-2), (org[0] + text_w + 2, org[1] + 5), bg_color, -1)
    img = cv2.putText (img, text, org, font, size, fg_color, thickness, linetype)
    return img

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
    return train_loader, test_loader, test_data

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
     
def eval_step(yolo, clip_model, clip_processor, data, device=get_device(), yolo_threshold=0.2, clip_threshold=0.0001):
    yolo.eval()   
    yolo_classes = get_yolo_classes()   
    clip_target, yolo_sentence = get_yolo_sentence(device)
    with torch.no_grad(): #important to keep memory free  
        for index in range(data.__len__()):
            #Init data
            cv2.destroyAllWindows() #new image, close previous windows
            input_img = data.__getimg__(index)
            info(input_img)
            CVimg = cv2.imread(input_img)
            PILimg = Image.open(input_img)
            
            #Compute YOLO predictions
            outputs_yolo = yolo(input_img)
            result = outputs_yolo.pandas().xyxy[0]

            #Compute CLIP predictions
            clip_inputs = clip_processor(PILimg).unsqueeze(0).to(device)
            logits_per_image, logits_per_textlip_outputs = clip_model(clip_inputs, clip_target)
            probs = logits_per_image.softmax(dim=1)
            top_probs, top_labels = probs.cuda().topk(5, dim=-1)

            #Draw bounding boxes for every probable class > threshold
            for i in range(0,4):
                if float(top_probs[0][i]) > clip_threshold:
                    #Draw YOLO bounding box
                    CVres = CVimg.copy()
                    color=(0,127,0)
                    yolo_found=False
                    for ind in result.index:
                        if result["confidence"][ind] > yolo_threshold and yolo_classes.index(result["name"][ind])==top_labels[0][i]:
                            yolo_found=True
                            cv2.rectangle (CVres, (int(result["xmin"][ind]), int(result["ymin"][ind])), (int(result["xmax"][ind]), int(result["ymax"][ind])), color, 4)
                    if yolo_found:
                        info(yolo_sentence[top_labels[0][i]] + " " + str(int(float(top_probs[0][i])*100))+"%")
                        info("Press ESC to exit program, any other key to continue")
                        CVres = putTextBg (CVres, yolo_sentence[top_labels[0][i]] + " " + str(int(float(top_probs[0][i])*100))+"%", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA, color)
                        cv2.imshow("Result", CVres)
                        if cv2.waitKey(0) == 27: #if you press ESC button, you will exit the program
                            return
           
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
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
    clip_model, clip_preprocess = clip.load("RN50", device=device)
    
    train_loader, test_loader, test_data = get_data(batch_size, annotations_file, root_imgs, clip_model, clip_preprocess, device, sample_size=num_samples)
    #plot_beginning(test_loader, clip_preprocess)
    info("Starting evaluation")
    eval_step(yolo_model, clip_model, clip_preprocess, test_data, device)
main()