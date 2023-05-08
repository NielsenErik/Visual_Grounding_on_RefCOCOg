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

def putTextBg(img, text, org, font, size, fg_color, thickness, linetype, bg_color):
    text_size, _ = cv2.getTextSize(text, font, size, thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, (org[0]-2, org[1]-text_h-2), (org[0] + text_w + 2, org[1] + 5), bg_color, -1)
    img = cv2.putText (img, text, org, font, size, fg_color, thickness, linetype)
    return img

YOLO_CLASSES={
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

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

def training_step(yolo, data_loader,  optimizer, cost_function = get_cost_function(), device=get_device(), yolo_threshold=0.5, clip_threshold=0.5):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    yolo.train()
    debugging("Training step")
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        input = inputs[0][0]
        outputs = yolo(input)
        debugging("Targets: " + str(targets))
        if len(targets)==0:
            info("EMPTY TARGETS IN TRAINING STEP")
            continue
        debugging("Training the network")
        #1 estrazione degli oggetti dalle immagini proposte da Yolo (seguendo i bounding boxes)
        CVimg = cv2.imread(input)
        PILimg = Image.open(input)
        result = outputs.pandas().xyxy[0]
        targets=[t[0] for t in targets]
        clip_targets = clip.tokenize(targets).to(device)
        # loss computation
        loss = torch.nn.CrossEntropyLoss(outputs, clip_targets)
        # backward pass
        loss.backward()        
        # parameters update
        optimizer.step()        
        # gradients reset
        optimizer.zero_grad()
            # fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1) # max() returns (maximum_value, index_of_maximum_value)
        # compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100
    
    pass

def test_step(yolo, clip_model, clip_processor, data_loader, device=get_device(), yolo_threshold=0.5, clip_threshold=0.5):
    yolo.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            input = inputs[0][0]
            outputs = yolo(input)
            targets=[t[0] for t in targets]
            debugging("Targets: " + str(targets))
            if len(targets)==0:
                info("EMPTY TARGETS")
                continue

            #1 estrazione degli oggetti dalle immagini proposte da Yolo (seguendo i bounding boxes)
            CVimg = cv2.imread(input)
            PILimg = Image.open(input)
            result = outputs.pandas().xyxy[0]
            clip_targets = clip.tokenize(targets).to(device)
            for ind in result.index:
                debugging("Object: " + str(result["name"][ind]) + " - Confidence: " + str(result["confidence"][ind]))
                if result["confidence"][ind] > yolo_threshold:
                    #2 foreach oggetto: valutazione similarità oggetto_ritagliato-target con clip (https://huggingface.co/docs/transformers/model_doc/clip#:~:text=from%20PIL%20import%20Image%0A%3E%3E%3E%20import%20requests%0A%0A%3E%3E%3E,take%20the%20softmax%20to%20get%20the%20label%20probabilities)
                    PILcropped = PILimg.crop((int(result["xmin"][ind]), int(result["ymin"][ind]), int(result["xmax"][ind]), int(result["ymax"][ind])))
                    #clip_inputs = clip_processor(text=targets, images=PILcropped, return_tensors="pt", padding=True)
                    clip_inputs = clip_processor(PILcropped).unsqueeze(0).to(device)
                    logits_per_image, logits_per_textlip_outputs = clip_model(clip_inputs, clip_targets)
                    #logits_per_image = clip_outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                    #3 prendere bounding box con massimo score di clip (e maggiore di determinato threshold) e visualizzare l'immagine con solo quella bounding box
                    top_probs, top_labels = probs.cuda().squeeze().topk(1)
                    debugging("Top probs: " + str(top_probs) + "Top probs type: " + str(type(top_probs)) + "Top probs shape: " + str(top_probs.shape) + " Top probs shape len: " + str(len(top_probs.shape)))
                    if len(top_probs.shape)==0:
                        debugging("EMPTY TOP PROBS")
                        continue
                    else:
                        CVres = CVimg.copy()
                        bgcolor = (0,127,0) if float(top_probs[0]) > clip_threshold else (0,0,127)
                        rectcolor = (0,255,0) if float(top_probs[0]) > clip_threshold else (0,0,255)
                        cv2.rectangle (CVres, (int(result["xmin"][ind]), int(result["ymin"][ind])), (int(result["xmax"][ind]), int(result["ymax"][ind])), rectcolor, 4)
                        CVres = putTextBg (CVres, str(targets[top_labels[0]]) + " " + str(int(float(top_probs[0])*100))+"%", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA, bgcolor)
                        debugging(str(targets[top_labels[0]]))
                        cv2.imshow("result", CVres)
                    cv2.waitKey(0)

##next step: https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb
            


batch_size = 1
device = 'cuda:0'
cost_function = get_cost_function()
learning_rate = 0.001
weight_decay = 0.000001
momentum = 0.9
epochs = 50,
num_classes = 65,
annotations_file = 'refcocog/annotations/refs(umd).p'
root_imgs = 'refcocog/images'

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
clip_model, clip_processor = clip.load('RN50', device=device)
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

optimizer = get_optimizer(yolo_model, learning_rate, weight_decay, momentum)

train_loader, test_loader = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs, model=clip_model, sample_size=50)
test_step(yolo_model, clip_model, clip_processor, test_loader, clip_threshold=0.8)

# train_loss, train_accuracy = training_step(yolo_model, train_loader, optimizer, cost_function)
# test_step(yolo_model, clip_model, clip_processor, test_loader, clip_threshold=0.8)
