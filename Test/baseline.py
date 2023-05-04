import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

import clip

#Here useful github page were there is an implementation
# of the baseline idea YOLO + Clip
# https://github.com/vijishmadhavan/Crop-CLIP/blob/master/Crop_CLIP.ipynb
# https://stackoverflow.com/questions/73593712/calculating-similarities-of-text-embeddings-using-clip

#import YOLO model

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
clip_model, clip_preprocess = clip.load("RN50", device=get_device())