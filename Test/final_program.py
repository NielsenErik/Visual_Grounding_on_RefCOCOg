from customClip import CustomClip
import torch
import cv2
import random
from pathlib import Path
from printCalls import info, debugging, error, warning

get_device_first_call=True
def get_device():
    global get_device_first_call
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if get_device_first_call:
        info("The current device is " + device)
        get_device_first_call=False
    return device

def main():
    clip_model = CustomClip(device=get_device())
    list_of_img = [img for img in Path("refcocog\images").glob('*')]

    while True:
        filename = random.choice(list_of_img)
        info(str(filename))
        img = cv2.imread(str(filename))
        cv2.imshow("Input img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        info("Insert your sentence")
        item = clip_model.__get_boxes_v2__(filename, input())
        if item is not None:
            cv2.rectangle(img, (item["xmin"], item["ymin"]), (item["xmax"], item["ymax"]), (0,127,0), 3)
            cv2.imshow("Result boxes", img)
            if cv2.waitKey(0) == 27: #if you press ESC button, you will exit the program
                cv2.destroyAllWindows()
                return
        
if __name__ == "__main__":
    main()