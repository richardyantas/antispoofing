from PIL import Image
from matplotlib.pyplot import imshow
import cv2
import pathlib
import smart_open
import torch
from torchvision import transforms
from pathlib import Path
import json
import argparse
from biometrics.src.data.casia import CASIA, crop_center_square

IMG_SIZE = 256
resize = 256

VIDEO_PATH = ""

def read_any_image(video, pos):
    cap = cv2.VideoCapture(f"{VIDEO_PATH}.avi")   # /dev/video0
    it = 0
    # ret, frame = cap.read()        
    # pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resize=(IMG_SIZE, IMG_SIZE)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(frame)
        #pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        it += 1
        if(it == pos):
            return pil_img
    return pil_img 


img_pil1 = read_any_image(1,5)
img_pil2 = read_any_image(3,5)
img_pil3 = read_any_image(5,5)
img_pil4 = read_any_image(7,5)

# img1 = cv2.resize(img_pil1, resize)
# img2 = cv2.resize(img_pil2, resize)
# img3 = cv2.resize(img_pil3, resize)
# img4 = cv2.resize(img_pil4, resize)

img_pil1.save("real.png")
img_pil2.save("printattack.png")
img_pil3.save("maskattack.png")
img_pil4.save("replayattack.png")

# img_pil.show()
# transform = transforms.Compose(
#              [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

