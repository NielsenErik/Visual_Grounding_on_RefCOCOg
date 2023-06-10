import torch
import math
from torchmetrics.classification import MulticlassRecall
from torchvision.ops import box_iou
from printCalls import info
from customClip import CustomClip
from cocoLoad import RefCOCO
from model_utilis import TensorBoard, save_model, load_model, putTextBg

get_device_first_call=True
def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def recall(preds, target, num_labels, device = get_device()):
    # Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
    # In the case of multiclass classification, the recall is the average of the recall of each class.
    # In the case of multilabel classification, the recall is the average of the recall of each label.
    # Parameters:
    # preds (torch.Tensor): Predicted values
    # target (torch.Tensor): Ground truth values
    # num_labels (int): Number of labels
    recall = MulticlassRecall(num_classes=num_labels, average=None).to(device)
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

def cosine_similarity(logits_img, logits_txt):
    # Semantic similarity is a metric defined over a set of documents or terms,
    # where the idea of distance between them is based on the likeness of their meaning or semantic content
    # Parameters:
    # custom_model (torch.nn.Module): custom model
    # img (string): Image
    # text (string): Description
    cos_sim = torch.nn.CosineSimilarity()
    similarity = cos_sim(logits_img, logits_txt)
    return similarity

def eval_step(model, eval_loader, device = get_device()):
    samples = 0.0
    cumulative_accuracy = 0.0
    cumulative_recall = 0.0
    cumulative_sim = 0.0
    cumulative_accuracy = 0.0
    model.eval() 
    model.float()
    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the set
        for (images, texts) in eval_loader:
            images = images.to(device)
            texts = texts.squeeze(1).to(device)
            logits_per_image, logits_per_texts = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            samples += images.shape[0]  
            n_labels = logits_per_texts.shape[1]
            _, predicted = logits_per_image.max(dim=1)
            cumulative_recall += torch.sum(recall(predicted, ground_truth, n_labels, device)).item()
            cumulative_accuracy += predicted.eq(ground_truth).sum().item()
            cumulative_sim += torch.sum(cosine_similarity(logits_per_image, logits_per_texts)).item()

    return cumulative_accuracy / samples, cumulative_recall / samples, cumulative_sim / samples


clip_model_ = CustomClip(device=get_device())
_, preprocess = clip_model_.__get_model__()
clip_model, epoch, loss = load_model(clip_model_, "Personal_Model/personal_model.pt")
test_data = RefCOCO(annotations_file = 'refcocog/annotations/refs(umd).p', img_dir='refcocog/images', preprocess = preprocess, split_type='test', device=get_device(), sample_size=100)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

# Evaluate recall (grounding accuracy metric) and cosine similarity (semantic similarity metric)
info("EVALUATING...")
acc, rec, sim = eval_step(clip_model, test_loader)
angle = math.degrees(math.acos(sim))
info("ACCURACY: {:2.1%} RECALL: {:2.1%} SIMILARITY: {:.4} -> {:.1f}°".format(acc, rec, sim, angle))


# Evaluate IoU (localization accuracy metric)
# IoU = 0
# for i in range(10):
#     _, img = test_data.__getimg__(random.randint(0, test_data.__len__()))
#     cvimg = cv2.imread(img)
#     cv2.imshow("Input",cvimg)
#     cv2.waitKey(0)
#     info("Insert sentence:")
#     text=input()
#     item, _ = clip_model.__get_boxes__(img, text)
#     if item is not None:
#         cv2.rectangle(cvimg, (item["xmin"], item["ymin"]), (item["xmax"], item["ymax"]), (0,127,0), 3)
#         cv2.imshow("Output",cvimg)
#         cv2.waitKey(0)

# info("IoU: {:2.1%}".format(IoU))
