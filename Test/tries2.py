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

YOLO_CLASSES=[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]
YOLO_SENTENCE = []
for el in YOLO_CLASSES:
    YOLO_SENTENCE.append("This image contains a "+el)
CLIP_TARGETS = clip.tokenize(YOLO_SENTENCE).to(get_device())

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

def test_step(yolo, clip_model, clip_processor, data_loader, device=get_device(), yolo_threshold=0.5, clip_threshold=0.5):
    yolo.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            pass

def eval_step(yolo, clip_model, clip_processor, data, device=get_device(), yolo_threshold=0.2, clip_threshold=0.2):
    yolo.eval()     
    with torch.no_grad(): #important to mantain memory free  
        for index in range(data.__len__()):
            #Init data
            input_img = data.__getimg__(index)
            info(input_img)
            CVimg = cv2.imread(input_img)
            PILimg = Image.open(input_img)
            
            #Compute YOLO predictions
            outputs_yolo = yolo(input_img)
            result = outputs_yolo.pandas().xyxy[0]

            #Compute CLIP predictions
            clip_inputs = clip_processor(PILimg).unsqueeze(0).to(device)
            logits_per_image, logits_per_textlip_outputs = clip_model(clip_inputs, CLIP_TARGETS)
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
                        if result["confidence"][ind] > yolo_threshold and YOLO_CLASSES.index(result["name"][ind])==top_labels[0][i]:
                            yolo_found=True
                            cv2.rectangle (CVres, (int(result["xmin"][ind]), int(result["ymin"][ind])), (int(result["xmax"][ind]), int(result["ymax"][ind])), color, 4)
                    if yolo_found:
                        info(YOLO_SENTENCE[top_labels[0][i]] + " " + str(int(float(top_probs[0][i])*100))+"%")
                        warning("Press ESC to exit program, any other key to continue")
                        CVres = putTextBg (CVres, YOLO_SENTENCE[top_labels[0][i]] + " " + str(int(float(top_probs[0][i])*100))+"%", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA, color)
                        cv2.imshow("Result", CVres)
                        if cv2.waitKey(0) == 27: #if you press ESC button, you will exit the program
                            return

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

train_loader, test_loader, test_data = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs, model=clip_model, sample_size=50)
eval_step(yolo_model, clip_model, clip_processor, test_data)

# train_loss, train_accuracy = training_step(yolo_model, train_loader, optimizer, cost_function)
# test_step(yolo_model, clip_model, clip_processor, test_loader, clip_threshold=0.8)

