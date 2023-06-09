import torch
import os
import cv2
from torch.utils.tensorboard import SummaryWriter

def save_model(model, epoch, optimizer, total_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, path+"/personal_model_"+str(epoch)+".pt")

def load_model(model, path):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']
    return model, epoch, loss

class TensorBoard():
    def __init__(self, log_dir):
        rootdir = log_dir
        max=1
        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d) and file.startswith("exp"):
                num = int(file.replace("exp",""))
                if num > max:
                    max = num
        log_dir = log_dir+"/exp"+str(max+1)
        self.writer = SummaryWriter(log_dir=log_dir)
    def log_values(self, step, loss, accuracy, prefix):
        self.writer.add_scalar(f"{prefix}/loss", loss, step)
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)
    def close(self):
        self.writer.close()

def putTextBg(img, text, org, font, size, fg_color, thickness, linetype, bg_color):
    text_size, _ = cv2.getTextSize(text, font, size, thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, (org[0]-2, org[1]-text_h-2), (org[0] + text_w + 2, org[1] + 5), bg_color, -1)
    img = cv2.putText (img, text, org, font, size, fg_color, thickness, linetype)
    return img
