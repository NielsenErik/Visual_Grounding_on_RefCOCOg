import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from cocoLoad import RefCOCO, RefCOCO_Split #Importing REfCOCO class from cocoLoad.py
from clip import clip
from printCalls import debugging, step

#

# get the cost function
def get_cost_function():
    return torch.nn.CrossEntropyLoss()

# get the optimization algorithm
def get_optimizer(net, lr, wd, momentum):
    return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

def cosine_similarity(image_z: torch.Tensor, texts_z: torch.Tensor):
    # normalise the image and the text
    print("image shape ", images_z.shape)
    print("text shape ", texts_z.shape)
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    # evaluate the cosine similarity between the sets of features
    similarity = (texts_z @ images_z.T)

    return similarity.cpu()

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
    
    transform = T.Compose(transform)
    return transform

def get_data(batch_size, annotations_file, img_root, model, preprocess, device = get_device(), sample_size = 5023):
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
    
    # In refCOCO there is already the plits inside the labels,
    # so we do't have to do the random split

    # refCOCO_data = RefCOCO(annotations_file = annotations_file, img_dir=img_root, transform=transform)
    # training_samples = int(num_samples * 0.8+1)
    # test_samples = num_samples - training_samples
    # training_data, test_data = torch.utils.data.random_split(
    #     refCOCO_data, [training_samples, test_samples])

    training_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='train', transform=transform, device=device, sample_size=sample_size)
    test_data = RefCOCO_Split(annotations_file = annotations_file, img_dir=img_root, model = model, preprocess = preprocess, split_type='test', transform=transform, device=device, sample_size=sample_size)
    num_training_samples = len(training_data)
    step("Number of training samples:" + str(num_training_samples))
    num_test_samples = len(test_data)
    step("Number of test samples:" + str(num_test_samples))

    # Number of training samples: 42226
    # Number of test samples: 5023
    
    # THis is only for test
    # it just get a smaller size of the training dataset
    
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def training_step(net, data_loader, optimizer, cost_function, device=get_device()):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs, targets)
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
    debugging("Into test function")
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            debugging("Into test loop")
            print(inputs.size(), targets.size())
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            res = outputs.pandas().xyxy[0]
            max_sim = 0
            max_sim_class = ""
            for ind in res.index:
                t = res["name"][ind]
                result = clip.tokenize(t).to(device)
                sim = cosine_similarity(result.float(), targets.float())
                if sim > max_sim:
                    max_sim = sim
                    max_sim_class = t
            print("The predicted class is", max_sim_class)
    #         debugging("Outputs sizes")
    #         print(outputs[0].size())
    #         print(outputs[1][0].size())
    #         print(outputs[1][1].size())
    #         print(outputs[1][2].size())
    #         loss = cost_function(outputs[0], targets)#NEW Bottleneck
    #         loss.backward()
    #         samples += inputs.shape[0]
    #         cumulative_loss += loss.item()
    #         _, predicted = outputs.max(dim=1)
    #         cumulative_accuracy += predicted.eq(targets).sum().item()
    # return cumulative_loss/samples, cumulative_accuracy/samples*100

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
device = get_device()

# import yolo baseline architecture
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, _verbose=False)
clip_model, clip_preprocess = clip.load('RN50', device=device)
clip_model = clip_model.cuda().eval()
train_loader, test_loader = get_data(batch_size, annotations_file=annotations_file, img_root=root_imgs, model=clip_model, preprocess=clip_preprocess, device=device, sample_size=100)

step("Before training")
train_loss, train_accuracy = test_step(clip_model, train_loader, get_cost_function(), device=device)
test_loss, test_accuracy = test_step(clip_model, test_loader, get_cost_function(), device=device)
step('\tTraining loss {:.5f}, Training accuracy {:.2f}'.format(
    train_loss, train_accuracy))
step('\tTest loss {:.5f}, Test accuracy {:.2f}'.format(
    test_loss, test_accuracy))
step('-----------------------------------------------------')
