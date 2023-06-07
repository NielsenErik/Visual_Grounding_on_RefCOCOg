from customClip import CustomClip
import torch
import cv2
import random
from pathlib import Path
from printCalls import info, debugging, error, warning
from cocoLoad import RefCOCO

get_device_first_call=True
def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def final_program(clip_model=None):
    if clip_model is None:
        clip_model = CustomClip(device=get_device())
    _, clip_processor = clip_model.__get_model__()
    sample_size = len([p for p in Path("refcocog/images").glob('*')])
    info("Total size: "+str(sample_size))
    test_data = RefCOCO(annotations_file = "refcocog/annotations/refs(umd).p", img_dir="refcocog\images", model=clip_model, preprocess=clip_processor, split_type='test', device=get_device(), sample_size=sample_size, batch_size=1)

    for i in range(100):
        #filename = random.choice(list_of_img)
        index = random.randint(0, test_data.__len__())
        _, image = test_data.__getimg__(index)
        textual_desc = random.choice(test_data.__gettext__(index))
        info("{:05d}: {} --> {}".format(index, image, textual_desc))

        img = cv2.imread(image)
        item = clip_model.__get_boxes_v2__(image, textual_desc)
        if item is not None:
            cv2.rectangle(img, (item["xmin"], item["ymin"]), (item["xmax"], item["ymax"]), (0,127,0), 3)
            cv2.imshow("Result boxes", img)
            if cv2.waitKey(0) == 27: #if you press ESC button, you will exit the program
                return
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    final_program()
    cv2.destroyAllWindows()