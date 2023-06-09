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


def random_get_text(all_texts):
    small_list = []
    for i in range(1000):
        small_list.append(all_texts[np.random.randint(0,len(all_texts))])
    return small_list

def get_all_texts(annotations_file, smallTest=True):
    x = pd.read_pickle(annotations_file)
    img_texts = pd.DataFrame(x)
    all_texts = []
    for x in img_texts.iloc():
        if x['split']=='test' and len(x['sentences'][0]['raw'])>0:
            all_texts.append(x['sentences'][0]['raw'])
    #info("Number of texts: " + str(len(all_texts)))
    if smallTest:
        all_texts = random_get_text(all_texts)
    #info("Number of random sample: " + str(len(all_texts)))
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
#RUNNING are the following:
# CrossEntropyLoss()
# HingeEmbeddingLoss()
# SmoothL1Loss()

def get_optimizer(net, lr, wd, momentum):
    return torch.optim.Adadelta(net.parameters(), lr=lr, weight_decay = wd)
    #return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay = wd)

# NO Adam
# OK ASGD run 10
# OK SGD run 11 Best
# NO RMSprop run 12
# No Nadam run 13
# No LBFGS run 14
# OK Adadelta run 15 seem GOOOODDD
# OK SGD Big Data run 16
# OK Adadelta Big Data run 17


def get_img_transform():
    transform = list()
    transform.append(T.ToPILImage())
#     # resize each PIL image to 256 x 256
#     transform.append(T.Resize((256, 256)))
#    # transform.append(T.CenterCrop((224,224)))   
#     # convert Numpy to Pytorch Tensor    
    transform.append(T.ToTensor())      
    transform = T.Compose(transform)
    return transform

def get_data(batch_size, annotations_file, img_root, model, test_batch_size = 16, preprocess = None, device = get_device(), sample_size_train = 42226, sample_size_test = 5023, sample_size_val = 2573, augment_data_train=True):

    transform = get_img_transform()
    
    sample_size_train = sample_size_train if sample_size_train <= 42226 else 42226
    sample_size_test = sample_size_test if sample_size_test <= 5023 else 5023
    sample_size_val = sample_size_val if sample_size_val <= 2573 else 2573
    training_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='train', transform=transform, device=device, sample_size=sample_size_train, batch_size=batch_size, augment_data=augment_data_train)
    test_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='test', transform=transform, device=device, sample_size=sample_size_test, batch_size=test_batch_size)
    eval_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='val', transform=transform, device=device, sample_size=sample_size_val, batch_size=test_batch_size)

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

def empty_token(model, device):
    empty_desc = clip.tokenize("").to(device)
    model, _ = clip.load("RN50", device=device)
    with torch.no_grad():
            empty_token = model.encode_text(empty_desc).float()  
    return empty_token

def training_step(model, train_dataloader,  optimizer, cost_function=get_cost_function(), device=get_device()):
    #https://github.com/openai/CLIP/issues/83#:~:text=for%20epoch%20in%20range,convert_weights(model)
    cumulative_accuracy = 0.0
    cumulative_loss = 0.0
    samples = 0.0
    empty_desc = empty_token(model, device)
    clip.model.convert_weights(model)
    model.train()
    for (images, texts) in train_dataloader:
        optimizer.zero_grad()
        #images, texts = batch
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
        #debugging(str(cumulative_loss))    
        samples += images.shape[0]  
        _, predicted = logits_per_image.max(dim=1)    
        cumulative_accuracy += predicted.eq(ground_truth).sum().item()
        # if device == "cpu":
        #     optimizer.step()
        # else : 
        #     for p in model.model.parameters(): 
        #         p.data = p.data.float() 
        #         p.grad.data = p.grad.data.float() 
        
        clip.model.convert_weights(model)
    return cumulative_loss / samples, cumulative_accuracy / samples

def test_step(model, test_loader, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    comulative_recall = 0.0
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
            n_labels = logits_per_texts.shape[1]
            _, predicted = logits_per_image.max(dim=1)
            cumulative_accuracy += predicted.eq(ground_truth).sum().item()
            comulative_recall += recall(predicted, ground_truth, n_labels, device)

        
    return cumulative_loss / samples, cumulative_accuracy / samples, comulative_recall / samples

def get_texts(data, device = get_device()):
    text = []
    for index in range(data.__len__()):
        for desc in data.__gettext__(index):
            text.append(desc)
            clip_targets = clip.tokenize(text).squeeze().to(device)
    return text, clip_targets

def eval_step(clip_model, clip_processor, data, coco_desc, device = get_device()):   
    clip_threshold = 0.0005
    clip_targets = clip.tokenize(coco_desc).squeeze().to(device)
    with torch.no_grad(): #important to mantain memory free  
        for index in range(data.__len__()):
                input_img, cv_input = data.__getimg__(index)
                CVimg = cv2.imread(cv_input)
                PILimg = Image.open(input_img)
                clip_inputs = clip_processor(PILimg).unsqueeze(0).to(device)
                logits_per_image, _ = clip_model(clip_inputs, clip_targets)
                probs = logits_per_image.softmax(dim=1)
                top_probs, top_labels = probs.cuda().topk(5, dim=-1)
                print(top_probs)
                print(top_labels)
                for i in range(0,1):
                    if len(top_probs[0])>i and float(top_probs[0][i]) > clip_threshold:
                        CVres = CVimg.copy()
                        color=(0,127,0)
                        info(coco_desc[top_labels[0][i]] + " " + str(int(float(top_probs[0][i])*100))+"%")
                        CVres = putTextBg (CVres, coco_desc[top_labels[0][i]] + " " + str(int(float(top_probs[0][i])*100))+"%", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA, color)
                        cv2.imshow("Result", CVres)
                        if cv2.waitKey(0) == 27: #if you press ESC button, you will exit the program
                            return
                        
def final_step(clip_model):
    filename="refcocog/images/COCO_train2014_000000045049.jpg"
    boxes = clip_model.__get_boxes_v1__(filename, "there is a girl playing tennis")
    img = cv2.imread(filename)
    for item in boxes:
        cv2.rectangle(img, (item["xmin"], item["ymin"]), (item["xmax"], item["ymax"]), (0,127,0), 4)
    cv2.imshow("Result", img)
    cv2.waitKey(0)

def update_parameters(learning_rate, weight_decay, momentum, alpha):
    learning_rate = learning_rate * alpha
    weight_decay = weight_decay * alpha
    momentum = momentum * alpha
    alpha = alpha/(alpha+0.001)
    return learning_rate, weight_decay, momentum, alpha

def main():

    #DATASET PARAMS
    sample_size_train=30720
    sample_size_test=5120
    sample_size_val=256
    augment_data_train=False

    #TRAINING PARAMS
    batch_size = 128 #must be 16 due to lenght of clip_targets
    test_batch_size = 64
    
    #OPTIMIZER & LOSS PARAMS
    cost_function = get_cost_function()
    learning_rate = 0.002
    weight_decay = 0.000001
    momentum = 0.9

    epochs = 20

    e = math.exp(1)
    alpha = 1
    
    
    visualization_name='RefCOCOg'


    annotations_file = 'refcocog/annotations/refs(umd).p'
    root_imgs = 'refcocog/images'
    all_texts = get_all_texts(annotations_file)
    #yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    clip_model = CustomClip(device=get_device(), batch_size=batch_size, norm=False, bias=True)
    _ , clip_processor = clip_model.__get_model__()
    #clip_model, clip_processor = clip.load('RN50', device, jit=False)
    optimizer = get_optimizer(clip_model, learning_rate, weight_decay, momentum)

    train_loader, test_loader, val_loader = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs, model=clip_model, test_batch_size = test_batch_size, preprocess=clip_processor, sample_size_train=sample_size_train, sample_size_test=sample_size_test, sample_size_val=sample_size_val, augment_data_train=augment_data_train)

    
    tb = TensorBoard("run")
    
    info("BEFORE TRAINING...")
    loss, accuracy, recall = test_step(clip_model, train_loader, cost_function)
    info("Train - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Train")
    loss, accuracy, recall = test_step(clip_model, val_loader, cost_function)
    info("Validation - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Validation")
    loss, accuracy, recall = test_step(clip_model, test_loader, cost_function)
    info("Test - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Test")
    optimizer = get_optimizer(clip_model, learning_rate, weight_decay, momentum)    

    info("INIT TRAINING...")
    for ep in range(1, epochs+1):
        info("EPOCH "+str(ep)+":")
        loss, accuracy = training_step(clip_model, train_loader, optimizer, cost_function)
        if ep % 5 == 0:
            save_model(clip_model, ep, optimizer, loss, "Personal_Model")
        info("Train - LOSS: {:.4} ACCURACY: {:2.1%}% ".format(loss, accuracy))
        tb.log_values(ep, loss, accuracy, "Train")
        loss, accuracy, recall = test_step(clip_model, val_loader, cost_function)
        info("Validation - LOSS: {:.4} ACCURACY: {:2.1%}%".format(loss, accuracy))
        tb.log_values(ep, loss, accuracy, "Validation") 
        learning_rate, weight_decay, momentum, alpha = update_parameters(learning_rate, weight_decay, momentum, alpha)
        optimizer = get_optimizer(clip_model, learning_rate, weight_decay, momentum)

    info("AFTER TRAINING...")
    loss, accuracy, recall = test_step(clip_model, train_loader, cost_function)
    info("Train - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Train")
    loss, accuracy, recall = test_step(clip_model, val_loader, cost_function)
    info("Validation - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Validation")
    loss, accuracy, recall = test_step(clip_model, test_loader, cost_function)
    info("Test - LOSS: {:.4} ACCURACY: {:2.1%}% RECALL: {:2.1%}".format(loss, accuracy, recall))
    tb.log_values(epochs+1, loss, accuracy, "Test")
    tb.close()

    save_model(clip_model, epochs, optimizer, loss, "Personal_Model")

    info("EVALUATION...")
    model, optimizer, epoch, loss = load_model(clip_model, optimizer, "Personal_Model")
    #eval_step(model, clip_processor, val_loader, all_texts, device=device)
    final_program.final_program(model)

##########################################################################################
main()
