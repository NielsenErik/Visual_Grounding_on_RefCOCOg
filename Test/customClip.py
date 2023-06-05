import torch
from torch import nn
import torch.nn.functional as F
import clip
import pandas as pd
from printCalls import error, warning, debugging, info 

#https://github.com/openai/CLIP/issues/83

class BatchNorm2d(torch.nn.Module):
  def __init__(self, in_features, track_running_stats=False, affine=True, momentum=0.9):
    super().__init__()
    
    self.in_features = in_features
    self.track_running_stats = track_running_stats
    self.affine = affine
    self.momentum = momentum
    
    if self.affine:
      self.gamma = torch.nn.Parameter(torch.ones(self.in_features, 1))
      self.beta = torch.nn.Parameter(torch.zeros(self.in_features, 1))
    
    if self.track_running_stats:
      # register_buffer registers a tensor as a buffer that will be saved as part of the model
      # but which does not require to be trained, differently from nn.Parameter
      self.register_buffer('running_mean', torch.zeros(self.in_features, 1))
      self.register_buffer('running_std', torch.ones(self.in_features, 1))
  
  def forward(self, x):
    # transpose (N, C, H, W) to (C, N, H, W)
    x = x.transpose(0, 1)
    
    # store the shape
    c, bs, h, w = x.shape
    
    # collapse all dimensions except the 'channel' dimension
    x = x.contiguous().view(c, -1)
    
    # calculate batch mean
    mean = x.mean(dim=1).view(-1, 1)
    
    # calculate batch std
    std = x.std(dim=1).view(-1, 1)
    
    # keep running statistics (moving average of mean and std)
    if self.training and self.track_running_stats:
      with torch.no_grad():
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_std = self.momentum * self.running_std + (1 - self.momentum) * std
    
    # during inference time
    if not self.training and self.track_running_stats:
      mean = self.running_mean
      std = self.running_std
    
    # normalize the input activations
    x = (x - mean) / std
    
    # scale and shift the normalized activations
    if self.affine:
      x = x * self.gamma + self.beta
    
    return x.view(c, bs, h, w).transpose(0, 1)


class CustomClip(torch.nn.Module):
    def __init__(self, device, batch_size, norm=True, bias=False):
        super().__init__()
        self.device = device
        model, self.preprocess = clip.load('RN50', device=self.device, jit=False)
        self.model = model
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
        self.in_features = 1024
        self.out_features = 1024
        
        self.bias = bias
        self.norm = norm
        self.batch_size = batch_size
        self.bottleneck = self.set_bottleneck()
        #self.encoder = self.model.visual.float()
        #self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
    
    def set_bottleneck(self):
        layer = [
                    torch.nn.Linear(self.in_features, self.in_features // 2, bias=self.bias),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.in_features // 2, self.in_features // 2, bias=self.bias),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.in_features // 2, self.out_features, bias=self.bias),
                ]   
        bottleneck = torch.nn.Sequential(*layer).to(self.device)
        return bottleneck
    
    def __get_model__(self):
        return self.model, self.preprocess    
    
    def __get_boxes__(self, input_img, input_text):
        detections = self.detector(input_img).pandas().xyxy[0]

        max_sim=0
        text_t = clip.tokenize(input_text).to(self.device)
        with torch.no_grad():
          enc_text = self.model.encode_text(text_t).float()
        cos_sim = nn.CosineSimilarity()
        for item in detections["name"]:
          cl_t = clip.tokenize(item).to(self.device)
          with torch.no_grad():
            enc_cl = self.model.encode_text(cl_t).float()
          dist = cos_sim(enc_cl, enc_text).item()
          debugging("{} <-> {}: {:2.1%}".format(input_text, item, dist))
          if dist > max_sim:
             max_sim = dist
             max_sim_cl = item

        boxes = []
        for _, item in detections.iterrows():
           if item["name"] == max_sim_cl:
              boxes.append({"xmin": int(item["xmin"]),"xmax": int(item["xmax"]),"ymin": int(item["ymin"]),"ymax": int(item["ymax"]),"confidence_class": item["confidence"], "confidence_text":max_sim})

        return boxes
      
      
    def forward(self, x, y):
        # if self.norm:
        #     x = self.bn1(x)
        #     print("Normed")
        
        #x = self.encoder(x)
        #y = self.bottleneck(y)
        #x = self.bottleneck(x)
        x, y = self.model.forward(x, y)
        return x, y