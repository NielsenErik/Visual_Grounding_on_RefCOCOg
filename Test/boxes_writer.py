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

def final_program(clip_model):
    warnings.filterwarnings("ignore")
    # if clip_model is None:
    #     clip_model = CustomClip(device=get_device())
    _, clip_processor = clip_model.__get_model__()
    annotations_file = "refcocog/annotations/refs(umd).p"
    sample_size = 5000#len([p for p in Path("refcocog/images").glob('*')])
    info("Total size: "+str(sample_size))
    training_data = RefCOCO(annotations_file = annotations_file, img_dir="refcocog/images", preprocess=clip_processor, split_type='train', device=get_device(), sample_size=40000)
    test_data = RefCOCO(annotations_file = annotations_file, img_dir="refcocog/images", preprocess=clip_processor, split_type='test', device=get_device(), sample_size=sample_size)
    validation_data = RefCOCO(annotations_file = annotations_file, img_dir="refcocog/images", preprocess=clip_processor, split_type='val', device=get_device(), sample_size=2000)
    x = pd.read_pickle(annotations_file)
    x = pd.DataFrame(x)
    
    x["xmin"] = np.nan
    x["ymin"] = np.nan
    x["xmax"] = np.nan
    x["ymax"] = np.nan
    for index in range(test_data.__len__()):
        #filename = random.choice(list_of_img)
        _, image = test_data.__getimg__(index)
        textual_desc = random.choice(test_data.__gettext__(index))
        if image is None or textual_desc is None:
            continue
        try:
            item, prob = clip_model.__get_boxes__(image, textual_desc)
        except:
            continue
        info("Image: "+image + ", textual description: "+textual_desc+ ", probability: "+str(prob))
        file_name = image.replace("refcocog/images/COCO_train2014_","").replace(".jpg","")
        id = int(file_name)
        if item is not None :
            x.loc[x["image_id"]==id, "xmin"] = item["xmin"]
            x.loc[x["image_id"]==id, "ymin"] = item["ymin"]
            x.loc[x["image_id"]==id, "xmax"] = item["xmax"]
            x.loc[x["image_id"]==id, "ymax"] = item["ymax"]
            debugging("xmin: "+str(item["xmin"])+", ymin: "+str(item["ymin"])+", xmax: "+str(item["xmax"])+", ymax: "+str(item["ymax"]))
    for index in range(training_data.__len__()):
        #filename = random.choice(list_of_img)
        _, image = training_data.__getimg__(index)
        textual_desc = random.choice(training_data.__gettext__(index))
        if image is None or textual_desc is None:
            continue
        try:
            item, prob = clip_model.__get_boxes__(image, textual_desc)
        except:
            continue
        info("Image: "+image + ", textual description: "+textual_desc+ ", probability: "+str(prob))
        file_name = image.replace("refcocog/images/COCO_train2014_","").replace(".jpg","")
        id = int(file_name)
        if item is not None :
            x.loc[x["image_id"]==id, "xmin"] = item["xmin"]
            x.loc[x["image_id"]==id, "ymin"] = item["ymin"]
            x.loc[x["image_id"]==id, "xmax"] = item["xmax"]
            x.loc[x["image_id"]==id, "ymax"] = item["ymax"]
            debugging("xmin: "+str(item["xmin"])+", ymin: "+str(item["ymin"])+", xmax: "+str(item["xmax"])+", ymax: "+str(item["ymax"]))
    for index in range(validation_data.__len__()):
        #filename = random.choice(list_of_img)
        _, image = validation_data.__getimg__(index)
        textual_desc = random.choice(validation_data.__gettext__(index))
        if image is None or textual_desc is None:
            continue
        try:
            item, prob = clip_model.__get_boxes__(image, textual_desc)
        except:
            continue
        info("Image: "+image + ", textual description: "+textual_desc+ ", probability: "+str(prob))
        file_name = image.replace("refcocog/images/COCO_train2014_","").replace(".jpg","")
        id = int(file_name)
        if item is not None :
            x.loc[x["image_id"]==id, "xmin"] = item["xmin"]
            x.loc[x["image_id"]==id, "ymin"] = item["ymin"]
            x.loc[x["image_id"]==id, "xmax"] = item["xmax"]
            x.loc[x["image_id"]==id, "ymax"] = item["ymax"]
            debugging("xmin: "+str(item["xmin"])+", ymin: "+str(item["ymin"])+", xmax: "+str(item["xmax"])+", ymax: "+str(item["ymax"]))
    pd.DataFrame.to_csv(x, "refcocog/annotations/images_with_boxes.csv")
    pd.DataFrame.to_pickle(x, "refcocog/annotations/images_with_boxes.p")
        
if __name__ == "__main__":
    clip_model = CustomClip(device=get_device())
    clip_model, epoch, loss = load_model(clip_model, "Personal_Model/FINAL_MODEL_personal_model_Adadelta_60K-16-aug-sigmoid.pt")
    final_program(clip_model)
    cv2.destroyAllWindows()