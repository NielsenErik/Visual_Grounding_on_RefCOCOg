import random
import warnings

import cv2
import torch
from cocoLoad import RefCOCO
from customClip import CustomClip
from model_utilis import TensorBoard, load_model, putTextBg, save_model
from printCalls import debugging, error, info, warning

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
        clip_model = CustomClip(device=get_device(), norm=False)
    _, clip_processor = clip_model.__get_model__()
    sample_size = 100
    info("Total size: "+str(sample_size))
    test_data = RefCOCO(annotations_file = "refcocog/annotations/refs(umd).p", img_dir="refcocog/images", preprocess=clip_processor, split_type='test', device=get_device(), sample_size=sample_size)

    for i in range(100):
        #filename = random.choice(list_of_img)
        index = random.randint(0, test_data.__len__())
        _, image = test_data.__getimg__(index)
        textual_desc = random.choice(test_data.__gettext__(index))

        img = cv2.imread(image)
        item, prob = clip_model.__get_boxes__(image, textual_desc)
        if item is not None:
            
            cv2.rectangle(img, (item["xmin"], item["ymin"]), (item["xmax"], item["ymax"]), (0,127,0), 3)
            info("{:05d}: {} --> {}\n\033[92m{:2.1%}\033[0m".format(index, image, textual_desc, prob))
            img = putTextBg (img, textual_desc + " " + str(int(float(prob)*100))+"%", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, (0,127,0))
            cv2.imshow("Result boxes", img)
            if cv2.waitKey(0) == 27: #if you press ESC button, you will exit the program
                return
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    clip_model = CustomClip(device=get_device(), norm=False)
    clip_model, epoch, loss = load_model(clip_model, "Personal_Model/personal_model.pt")

    final_program(clip_model)
    cv2.destroyAllWindows()