from customClip import CustomClip
import torch
import cv2
import random
from printCalls import info, debugging, error, warning
from cocoLoad import RefCOCO
from model_utilis import TensorBoard, save_model, load_model, putTextBg
import pandas as pd
import numpy as np
import warnings


get_device_first_call=True
def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def final_program(clip_model=None):
    warnings.filterwarnings("ignore")
    if clip_model is None:
        clip_model = CustomClip(device=get_device())
    _, clip_processor = clip_model.__get_model__()
    annotations_file = "refcocog/annotations/refs(umd)_w_boxes.p"
    sample_size = 5000#len([p for p in Path("refcocog/images").glob('*')])
    info("Total size: "+str(sample_size))
    test_data = RefCOCO(annotations_file = annotations_file, img_dir="refcocog/images", preprocess=clip_processor, split_type='train', device=get_device(), sample_size=sample_size)
    x = pd.read_pickle(annotations_file)
    x = pd.DataFrame(x)
    x["xmin"] = np.nan
    x["ymin"] = np.nan
    x["xmax"] = np.nan
    x["ymax"] = np.nan
    for i in range(50):
        #filename = random.choice(list_of_img)
        index = random.randint(0, test_data.__len__()-1)
        _, image = test_data.__getimg__(i)
        textual_desc = random.choice(test_data.__gettext__(i))
        print(image," ", textual_desc)
        item, prob = clip_model.__get_boxes__(image, textual_desc)
        info("Image: "+image + ", textual description: "+textual_desc+ ", probability: "+str(prob))
        if item is not None:
            x["xmin"].iloc[index] = item["xmin"]
            x["ymin"].iloc[index] = item["ymin"]
            x["xmax"].iloc[index] = item["xmax"]
            x["ymax"].iloc[index] = item["ymax"]
            debugging("xmin: "+str(item["xmin"])+", ymin: "+str(item["ymin"])+", xmax: "+str(item["xmax"])+", ymax: "+str(item["ymax"]))
    pd.DataFrame.to_csv(x, "annotations.csv")
    pd.to_pickle(x, annotations_file)
        
if __name__ == "__main__":
    clip_model = CustomClip(device=get_device())
    clip_model, epoch, loss = load_model(clip_model, "Personal_Model/Model2.pt")
    final_program(clip_model)
    cv2.destroyAllWindows()