import torch
import clip

class CustomClip(torch.nn.Module):
    def __init__(self, device, bias=False):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load('RN50', self.device, jit=False)
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.in_features = 1024
        self.out_features = 1024
        self.bias = bias
        # take the visual encoder of CLIP
        # we also convert it to be 32 bit (by default CLIP is 16)
        #self.model_encoder = self.model.visual.float()
        self.bottleneck = self.set_bottleneck()
   
    def set_bottleneck(self):
        layer = [
                    torch.nn.Linear(self.in_features, self.in_features // 2, bias=self.bias),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.in_features // 2, self.in_features // 2, bias=self.bias),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.in_features // 2, self.out_features, bias=self.bias),
                ]   
        bottleneck = torch.nn.Sequential(*layer)
        return bottleneck
    
    def __get_model__(self):
        return self.model, self.preprocess    
    
    def __get_boxes__(self):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward()
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x