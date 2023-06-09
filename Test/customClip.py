import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import clip
from PIL import Image
from printCalls import error, warning, debugging, info

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
    def __init__(self, device, batch_size=128, norm=True, bias=True):
        super().__init__()
        self.device = device
        model, self.preprocess = clip.load('RN50', device=self.device, jit=False)
        self.model = model
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
        #self.encoder = self.model.visual.float()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        #self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
    
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
    
    def __get_boxes_v1__(self, input_img, input_text):
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
    
    def __get_boxes_v2__(self, input_img, input_text):
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

<<<<<<< HEAD
        return bounding_boxes[top_label.item()], top_prob.item()
      
    def forward(self, image, text):
        #image = self.encoder(image).to(self.device)       
        image = self.model.encode_image(image)
        with autocast(dtype=torch.half):
           image = self.img_bottleneck(image).to(self.device)
        text = self.model.encode_text(text)
        with autocast(dtype=torch.half):
           text = self.txt_bottleneck(text).to(self.device)
=======
        max_sim=0
        for index, img in enumerate(img_cropped):
          preprocessed_img = self.preprocess(img).unsqueeze(0).to(self.device)
          with torch.no_grad():
            input_clip_img = self.model.encode_image(preprocessed_img).float()
          dist = cos_sim(input_clip_img, input_clip_txt).item()
          if dist>max_sim:
             max_sim=dist
             max_sim_index = index
          
        return bounding_boxes[max_sim_index]
      
    def forward(self, image, text):
        # image = self.encoder(image).to(self.device)
        # image = self.img_bottleneck(image).to(self.device)
        image = self.model.encode_image(image).float()
        text = self.model.encode_text(text).float()
>>>>>>> 8b23d45 (removed momentanery bottleneck)

        if self.norm:
            image = self.bn1(image).to(self.device)  
            text = self.bn1(text).to(self.device)      
        
        image = image / image.norm(dim=-1, keepdim=True).float()
        text = text / text.norm(dim=-1, keepdim=True).float()
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image @ text.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text