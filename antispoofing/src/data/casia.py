
# from biometrics.src.models.cnn import IMAGE_SIZE
import glob
import json
import os
import argparse
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from biometrics.src.utils.util_data import BaseDataset, split_dataset
import cv2


DATA_DIR = Path(__file__).resolve().parents[3]  # / "storage/datasets/"
CONFIG_DIR = Path(__file__).resolve().parents[3]
CONFIG = json.load(open(f"{CONFIG_DIR}/config.json"))
IMG_SIZE = CONFIG["training_config"]["image_size"]  # 256
TRAIN_FRAC = 0.8

# DATA_DIR = "storage/datasets/CASIA_faceAntisp/"
# IMG_SIZE = 256  # 24


# def _download_CASIA_dataset():
#     if os.path.exists(f"{DATA_DIR}/CASIA_faceAntisp"):
#         print("file already uncompressed and exist")
#     elif os.path.exists(f"{DATA_DIR}/CASIA_faceAntisp.rar"):
#         print("Uncompressing...")
#         print("Done and uncompressed!", flush=True)
#     else:
#         url = "https://www.dropbox.com/s/aaz282d9wyst0w8/CASIA_faceAntisp.rar"
#         print("- Downloading Casia dataset... ", end='', flush=True)
#         filename, headers = urllib.request.urlretrieve(
#             url, f"{DATA_DIR}/CASIA_faceAntisp.rar")
#         # req = requests.get(url)
#         # filename = url.split('/')[-1]
#         # with open(f"{DATA_DIR}/{filename}", 'wb') as output_file:
#         #     output_file.write(req.content)
#         print("Done and uncompressed!", flush=True)
#     print("done!")


# def _extract_CASIA_dataset():
#     # with rarfile.RarFile(f"{DATA_DIR}/CASIA_faceAntisp.rar") as r:
#     #     r.extractall(f"{DATA_DIR}/CASIA_faceAntisp")
#     os.mkdir(f"{DATA_DIR}/CASIA_faceAntisp")
#     patoolib.extract_archive(
#         f"{DATA_DIR}/CASIA_faceAntisp.rar", outdir=f"{DATA_DIR}/CASIA_faceAntisp")
#     print("done!")


BATCH_SIZE = 12  # 128
NUM_WORKERS = 0


def _download_CASIA_dataset():
    if os.path.exists(f"{DATA_DIR}/CASIA_faceAntisp"):
        print("file already uncompressed and exist")
    elif os.path.exists(f"{DATA_DIR}/CASIA_faceAntisp.rar"):
        print("Uncompressing...")
        print("Done and uncompressed!", flush=True)
    else:
        url = "https://www.dropbox.com/s/aaz282d9wyst0w8/CASIA_faceAntisp.rar"
        print("- Downloading Casia dataset... ", end='', flush=True)
        filename, headers = urllib.request.urlretrieve(
            url, f"{DATA_DIR}/CASIA_faceAntisp.rar")
        # req = requests.get(url)
        # filename = url.split('/')[-1]
        # with open(f"{DATA_DIR}/{filename}", 'wb') as output_file:
        #     output_file.write(req.content)
        print("Done and uncompressed!", flush=True)
    print("done!")


def _extract_CASIA_dataset():
    # with rarfile.RarFile(f"{DATA_DIR}/CASIA_faceAntisp.rar") as r:
    #     r.extractall(f"{DATA_DIR}/CASIA_faceAntisp")
    os.mkdir(f"{DATA_DIR}/CASIA_faceAntisp")
    patoolib.extract_archive(
        f"{DATA_DIR}/CASIA_faceAntisp.rar", outdir=f"{DATA_DIR}/CASIA_faceAntisp")
    print("done!")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)  # // floor division
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("video processed")
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        if len(frames) == max_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)


def load_x16_frames_from_video(filename):
    frames = load_video(filename)
    # fra = frames.copy()
    step = len(frames)//16
    frames = frames[::step]

    if len(frames) == 19:
        frames = frames[:-2]
    if len(frames) == 16:
        print(":9999")
        #tmp = frames[-1]
        frames = list(frames)
        frames.append(frames[-1])
        frames = np.array(frames)
        # step = len(fra)//15
        # frames = fra[::step]
        # frames = np.append(frames, frames[0].copy())  # ??
        print("new size", len(frames))
    if len(frames) == 18:
        frames = frames[:-1]
    return frames


# mapl = {
#     "1": [0, 0, 0, 1],  # 1 real
#     "2": [0, 0, 0, 1],  # 1 real
#     "3": [0, 0, 1, 0],  # 2 print attack
#     "4": [0, 0, 1, 0],  # 2 print attack
#     "5": [0, 1, 0, 0],  # 3 print mask attack
#     "6": [0, 1, 0, 0],  # 3 print mask attack
#     "7": [1, 0, 0, 0],  # 4 replay attack
#     "8": [1, 0, 0, 0],   # 4 replay attack
#     "HR_1": [0, 0, 0, 1],
#     "HR_2": [0, 0, 1, 0],
#     "HR_3": [0, 1, 0, 0],
#     "HR_4": [1, 0, 0, 0]
# }

# mapl = {
#     "1": 1,
#     "2": 1,
#     "3": 2,
#     "4": 2,
#     "5": 3,
#     "6": 3,
#     "7": 4,
#     "8": 4,
#     "HR_1": 1,
#     "HR_2": 2,
#     "HR_3": 3,
#     "HR_4": 4
# }
mapl = {
    "1": 0,
    "2": 0,
    "3": 1,
    "4": 1,
    "5": 2,
    "6": 2,
    "7": 3,
    "8": 3,
    "HR_1": 0,
    "HR_2": 1,
    "HR_3": 2,
    "HR_4": 3
}


def load_dataset(path):  # path_train: storage/datasets/CASIA_faceAntisp/train_release/  , path_test = storage/datasets/CASIA_faceAntisp/test_release/
    NUM_FOLDERS = 1
    dataset_x = []
    dataset_y = []
    try:
        files = glob.glob(
            '/home/testing/temporal/CASIA/**/*.avi', recursive=True)
        for f in files:
            frames = load_x16_frames_from_video(f)
            # print(" tamanio nframes: ", len(frames))
            dataset_x = [*dataset_x, *frames]  # verificar
            dataset_x = np.array(dataset_x)
            # print("f: ", f)
            fn = f.split("/")
            for i in range(0, 17):
                dataset_y.append(mapl[fn[-1][:-4]])
    finally:
        dataset_x = dataset_x.reshape(-1, IMG_SIZE,
                                      IMG_SIZE).swapaxes(1, 2)
    return dataset_x, dataset_y


# def load_dataset(path):  # path_train: storage/datasets/CASIA_faceAntisp/train_release/  , path_test = storage/datasets/CASIA_faceAntisp/test_release/
#     NUM_FOLDERS = 1
#     dataset_x = []
#     dataset_y = []
#     owd = os.getcwd()
#     print(owd)
#     try:
#         os.chdir(path)
#         for folder in range(1, NUM_FOLDERS+1):
#             os.chdir(str(folder))
#             files = os.listdir()
#             for f in files:
#                 # print(f"folder: {folder} - video: {f}")
#                 # for each video
#                 if f.endswith('.avi'):
#                     frames = load_x16_frames_from_video(f)
#                     print(" tamanio nframes: ", len(frames))
#                     # frames = frames.flat[:]
#                     dataset_x = [*dataset_x, *frames]  # verificar
#                     # dataset_x = np.append(dataset_x, frames)
#                     dataset_x = np.array(dataset_x)
#                     # for each
#                     for i in range(0, 17):
#                         dataset_y.append(mapl[f[:-4]])
#                         # dataset_y.append(np.array(mapl[f[:-4]], np.int_))  #§
#                         # dataset_y.append(np.array(mapl[f[:-4]], np.int_))  #
#                         #   dataset_y.append(np.array(mapl[f[:-4]], np.float_))  # does not work
#                         #label = mapl[f[:-4]]
#                         #dataset_y = [*dataset_y, *label]
#             os.chdir("..")
#     finally:
#         os.system(owd)
#         # dataset_x = dataset_x.reshape(-1, 28, 28).swapaxes(1, 2)
#         dataset_x = dataset_x.reshape(-1, IMG_SIZE,
#                                       IMG_SIZE).swapaxes(1, 2)
#         # dataset_x = np.ndarray(dataset_x)
#         # dataset_x = dataset_x.flatten()
#         # return dataset_x, torch.tensor(dataset_y, dtype=torch.float)
#     return dataset_x, dataset_y

class CASIA(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # self.mapping = ["real", "printattack",
        #                 "replayattack", "printmaskattack"]
        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = transforms.Compose(
            [transforms.ToTensor()])
        self.target_transform = transforms.Compose(
            [transforms.ToTensor()])

        self.dims = (1, IMG_SIZE, IMG_SIZE)
        # self.dims = (1, 640, 640)  # (1, *essentials["input_shape"])
        # self.output_dims = (1,)
        # self.output_dims = (4,)
        self.mapping = list(range(4))
        self.bath_size = 50
        self.train_path = f"{DATA_DIR}/CASIA/train_release"
        self.test_path = f"{DATA_DIR}/CASIA/test_release"

        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[4] / "datasets"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        # return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}
        return {"input_dims": self.dims, "mapping": self.mapping}

    # def prepare_data(self, *args, **kwargs) -> None:
    #     """
    #     Use this method to do things that might write to disk or that need to be done only from a single GPU
    #     in distributed settings (so don't set state `self.x = y`).
    #     """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.x_trainval, self.y_trainval = load_dataset(
            f"{DATA_DIR}/CASIA/train_release")
        print("data loaded ..")

        # training data set
        data_trainval = BaseDataset(
            self.x_trainval, self.y_trainval, transform=self.transform, target_transform=None)  # self.target_transform

        x0_train, y0_train = data_trainval[0]

        print(x0_train)
        print(y0_train)

        self.data_train, self.data_val = split_dataset(
            base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)
        self.x_test, self.y_test = load_dataset(
            f"{DATA_DIR}/CASIA/test_release")

        # testing data set
        self.data_test = BaseDataset(
            self.x_test, self.y_test, transform=self.transform, target_transform=None)  # self.target_transform

    def __repr__(self):
        basic = f"CASIA Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
