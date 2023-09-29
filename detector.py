from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import transforms
from pathlib import Path
import cv2
import pathlib
import smart_open
import torch
import json
import argparse

#CONFIG_DIR = Path(__file__).resolve().parents[0]
#CONFIG = json.load(open(f"{CONFIG_DIR}/config.json"))
#IMG_SIZE = CONFIG["training_config"]["image_size"]
#IMG_SIZE = 256
VIDEO_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = pathlib.Path(__file__).resolve().parent
# video_path=f"{VIDEO_DIR}/CASIA/test_release/1/3.avi"


class AntiSpoofingDetector:
    def __init__(self) -> None:
        lit_model = torch.load(
            "{MODEL_DIR}/model.pt")
        lit_model.eval()
        self.scripted_model = lit_model.to_torchscript(
            method="script", file_path=None)

    def predict(self, image_path):
        res = 256
        img = cv2.imread(image_path)
        print("type: ", type(img))
        img = cv2.resize(img, (256, 256))
        #print("shape: ", img.shape())
        img_pil = Image.fromarray(img)
        img_pil = img_pil.convert(mode="L")  # grayscale convertion
        img_pil.show()
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img_pil)
        print(img_tensor)
        # cv2.imshow(img)
        y_pred = self.scripted_model(img_tensor.unsqueeze(axis=0))[0]
        # print(y_pred)
        return y_pred


def main():
    # """
    # This file is only for testing the model with an image
    # Example runs:
    # ```
    # python detector.py example.png
    # python detector.py https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
    # """
    parser = argparse.ArgumentParser(
        description="Recognize handwritten text in an image file.")
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    detector = AntiSpoofingDetector()
    pred_str = detector.predict(args.filename)
    print(pred_str)
    _, yhat = torch.max(pred_str, -1)
    print("after max: ", yhat)


if __name__ == "__main__":
    main()
