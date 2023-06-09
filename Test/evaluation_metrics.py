import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from torchmetrics.classification import MulticlassRecall
from torchvision.ops import box_iou

def recall(preds, target, num_labels, device = 'cuda'):
    # Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
    # In the case of multiclass classification, the recall is the average of the recall of each class.
    # In the case of multilabel classification, the recall is the average of the recall of each label.
    # Parameters:
    # preds (torch.Tensor): Predicted values
    # target (torch.Tensor): Ground truth values
    # num_labels (int): Number of labels
    recall = MulticlassRecall(num_classes=num_labels).to(device)
    multilabel_recall = recall(preds, target)
    return multilabel_recall

def intersection_over_union(preds, target):
    # Intersection over union is a measure of the overlap between two bounding boxes.
    # It is calculated by dividing the area of overlap by the area of union.
    # Parameters:
    # preds (dictionary contains xmin, xmax, ymin, ymax): Predicted values
    # target (dictionary contains xmin, xmax, ymin, ymax): Ground truth values
    preds = torch.Tensor([preds["xmin"], preds["ymin"], preds["xmax"], preds["ymax"]])
    target = torch.Tensor([target["xmin"], target["ymin"], target["xmax"], target["ymax"]])
    value = box_iou(preds, target)
    return value.item()

def semantic_similarity(clip_model, clip_preprocess, img, text, device = 'cuda'):
    # Semantic similarity is a metric defined over a set of documents or terms,
    # where the idea of distance between them is based on the likeness of their meaning or semantic content
    # Parameters:
    # clip_model (torch.nn.Module): CLIP model
    # img (string): Image
    # text (string): Description
    text = clip.tokenize(text).to(device)
    img = clip_preprocess(Image.open(img))
    clip_model.eval()
    with torch.no_grad():
        enc_text = clip_model.encode_text(text).float()
        enc_img = clip_model.encode_image(img).float()
    cos_sim = torch.nn.CosineSimilarity()
    similarity = cos_sim(enc_img, enc_text)
    return similarity.item()


if __name__ == "__main__":
    pass