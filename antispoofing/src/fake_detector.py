import time
import logging
import imutils
import cv2
import torch
import numpy as np

from array import array
from torchvision import transforms
from pathlib import Path
from biometrics.src.utils.common import *

logger = logging.getLogger(__name__)
MODEL_PATH = f"{ROOT_PATH}/model.pt"


class FaceLiveness_COLORSPACE_YCRCBLUV:

    def __init__(self, path):
        self.lit_model = torch.load(MODEL_PATH)
        self.lit_model.eval()
        self.scripted_model = self.lit_model.to_torchscript(
            method="script", file_path=None)

    # def predict_from_image(self, image):
        # img_pil = Image.open(image)
        # print("shape: ", img_pil.size)
        # img_pil = img_pil.convert(mode="L")  # grayscale convertion
        # img_pil.show()

    def predict(self, frame):
        # frame = crop_center_square(frame)
        resize = (256, 256)
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(frame)
        print(img_tensor)
        y_pred = self.scripted_model(img_tensor.unsqueeze(axis=0))[0]
        return y_pred

    def is_fake(self, frame):
        _, res = torch.max(self.predict(frame), -1)
        # self.lit_model.clear()
        if res == 1:
            print("real")
            return False
        elif res == 2:
            print("print attack")
            return True
        elif res == 3:
            print("mask attack")
            return True
        else:
            print("replay attack")
            return True


fake_detector = FaceLiveness_COLORSPACE_YCRCBLUV('storage/weights/')


def detect_profile(face_cascade, gray_image):
    profile_detected = False
    imaged_flipped = False
    nose_left = False
    while profile_detected is False:
        profiles = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in profiles:
            if x > 0 and y > 0 and w > 0 and h > 0:
                profile_detected = True
        if imaged_flipped and profile_detected:
            nose_left = True
        if imaged_flipped:
            break
        gray_image = cv2.flip(gray_image, 1)
        imaged_flipped = True
    if profile_detected and nose_left is False:
        return True
    elif profile_detected and nose_left is True:
        return True, True
    else:
        return False


def detect_and_classify(imgpath: Path):
    start_time = time.time()
    imgpath = str(imgpath)
    logger.info('Opening video...')
    video = cv2.VideoCapture(imgpath)
    face_cascade_central = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade_profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
    # print("Reading video .. ")
    logger.info('Reading video...1')
    # fake_dectector = FaceLiveness_COLORSPACE_YCRCBLUV('utils/trained_models/')
    # fake_detector = FaceLiveness_COLORSPACE_YCRCBLUV('storage/weights/')

    logger.info('model loaded...')
    pos_frame = video.get(1)
    # we make an array of results for each frame to see if there's a legitimate face or not
    overall_results = []
    # we are going to stablish as a factor de movement of the head side to side and capture the profile frames
    # define  an array of size 2 -> 'b': int type

    #best_frames = array('b', [0, 0])
    best_frames = array('i', [0, 0])

    head_movement = False
    while True:
        ret, frame = video.read()
        if ret:
            frame = imutils.resize(frame, width=320, height=240)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detected = face_cascade_central.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
            scan_profile = detect_profile(face_cascade_profile, gray)
            if isinstance(scan_profile, bool) and scan_profile is True:
                # best_frames['profile_nose_left_frames'].append(frame)
                best_frames[0] += 1
            elif isinstance(scan_profile, tuple):
                # best_frames['profile_nose_right_frames'].append(frame)
                best_frames[1] += 1
            if scan_profile is not True and not isinstance(scan_profile, tuple):
                if isinstance(face_detected, np.ndarray):
                    # (x, y, w, h) = face_detected[0]
                    is_fake_video = fake_detector.is_fake(frame)
                    if is_fake_video:
                        overall_results.append({'is_real': False})
                    else:
                        overall_results.append({'is_real': True})
        else:
            video.set(1, pos_frame - 1)
        if video.get(1) == video.get(7):
            break
    if sum(best_frames)/2 > 10.0:
        head_movement = True
    cv2.destroyAllWindows()
    video.release()
    # print("Profile frames- ")
    logger.info('profile frames - ' +
                str(best_frames[0]) + ' ' + str(best_frames[1]))
    is_real_check = (len([frame for frame in overall_results if frame['is_real'] is True]) /
                     len(overall_results)) * 50  # this check equals half of the evaluation
    logger.info('real factor - ' + str(is_real_check))
    logger.info('overall_central_frames -' + str(len(overall_results)))
    theres_movement = int(head_movement) * 50  # this equals the other half
    logger.info('movement factor -' + str(theres_movement))
    # print("Video processed! ")
    logger.info('Video processed!')
    end_time = time.time()
    total_processing_time = end_time - start_time
    #print(f'total_processing_time {total_processing_time}')
    logger.info(f'total_processing_time {total_processing_time}')
    return is_real_check + theres_movement
