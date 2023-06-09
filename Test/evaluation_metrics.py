import torch
import numpy as np
import pandas as pd
from torchmetrics.classification import MulticlassRecall

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

def intersection_over_union():
    pass

def semantic_similarity():
    pass