import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
import clip
from PIL import Image
from printCalls import error, warning, debugging, info
from model_utilis import load_model

#https://github.com/openai/CLIP/issues/83

class BatchNorm1d(torch.nn.Module):
  def __init__(self, in_features, track_running_stats=True, affine=True, momentum=0.9, device = 'cuda:0'):
    super().__init__()
    
    self.in_features = in_features
    self.track_running_stats = track_running_stats
    self.affine = affine
    
    self.device = device
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
    # transpose (N, C) to (C, N)
    x = x.to(self.device)
    x = x.transpose(0, 1).contiguous().view(x.shape[1], -1).to(self.device)
    
    # calculate batch mean
    mean = x.mean(dim=1).view(-1, 1).to(self.device)
    
    # calculate batch std
    std = x.std(dim=1).view(-1, 1).to(self.device)
    
    # during training keep running statistics (moving average of mean and std)
    if self.training and self.track_running_stats:
      # no computational graph is necessary to be built for this computation
      with torch.no_grad():
        self.running_mean = (self.momentum * self.running_mean + (1 - self.momentum) * mean)
        self.running_std = (self.momentum * self.running_std + (1 - self.momentum) * std)
    
    # during inference time
    if not self.training and self.track_running_stats:
      mean = self.running_mean
      std = self.running_std
    
    # normalize the input activations
    x = (x - mean) / std
    
    # scale and shift the normalized activations
    if self.affine:
      x = x * self.gamma.to(self.device) + self.beta.to(self.device)

    return x.transpose(0, 1)


class CustomClip(torch.nn.Module):
    def __init__(self, device, custom_model_path=None, batch_size=128, norm=True, bias=True):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load('RN50', device=self.device, jit=False)
        if custom_model_path is not None:
          self.model, _, _ = load_model(self.model, custom_model_path)
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
        self.in_features = 1024
        self.out_features = 1024
        self.norm = norm
        if self.norm:
          self.bn1 = BatchNorm1d(self.in_features, track_running_stats=False, affine=False, momentum=0.9)
        self.bias = False
        self.norm = norm
        self.batch_size = batch_size
        self.img_bottleneck = self.set_img_bottleneck()
        self.txt_bottleneck = self.set_txt_bottleneck()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def set_img_bottleneck(self):
        layer = [
                    torch.nn.Linear(self.in_features, self.in_features // 2, bias=self.bias),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.in_features // 2, self.in_features // 2, bias=self.bias),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.in_features // 2, self.out_features, bias=self.bias),
                ]   
        bottleneck = torch.nn.Sequential(*layer).to(self.device)
        return bottleneck
    
    def set_txt_bottleneck(self):
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
        self.model.eval()
        detections = self.detector(input_img).pandas().xyxy[0]
        image = Image.open(input_img)
        img_cropped = []
        bounding_boxes = []
        for _, item in detections.iterrows():
          xmin = int(item["xmin"])-30 if int(item["xmin"])-30 >= 0 else 0
          ymin = int(item["ymin"])-30 if int(item["ymin"])-30 >= 0 else 0
          xmax = int(item["xmax"])+30 if int(item["xmax"])+30 <= image.size[0] else image.size[0]
          ymax = int(item["ymax"])+30 if int(item["ymax"])+30 <= image.size[1] else image.size[1]
          cropped = image.crop((xmin, ymin, xmax, ymax))
          img_cropped.append(cropped)
          bounding_boxes.append({"xmin": int(item["xmin"]), "ymin": int(item["ymin"]), "xmax": int(item["xmax"]), "ymax": int(item["ymax"])})

        if(len(img_cropped)==0):
           return None

        with torch.no_grad():
          tokenized_text = clip.tokenize([input_text]).squeeze(1).to(self.device)

          preprocessed_imgs=[]
          for index, img in enumerate(img_cropped):
            preprocessed_imgs.append(self.preprocess(img).to(self.device))
          preprocessed_imgs = torch.stack(preprocessed_imgs)
          self.model.float()
          _, logits_per_text = self.model(preprocessed_imgs, tokenized_text)
          probs = logits_per_text.softmax(dim=1)
          top_prob, top_label = probs.topk(1, dim=-1)

        return bounding_boxes[top_label.item()], top_prob.item()
      
    def forward(self, image, text):
        #image = self.encoder(image).to(self.device)       
        image = self.model.encode_image(image)
        with autocast(dtype=torch.half):
           image = self.img_bottleneck(image).to(self.device)
        text = self.model.encode_text(text)
        with autocast(dtype=torch.half):
           text = self.txt_bottleneck(text).to(self.device)


        if self.norm:
            image = self.bn1(image).to(self.device)  
            text = self.bn1(text).to(self.device)      
        
        image = image / image.norm(dim=-1, keepdim=True).float()
        text = text / text.norm(dim=-1, keepdim=True).float()
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image @ text.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text