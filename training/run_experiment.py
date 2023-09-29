import logging
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import argparse
import importlib
from logging.handlers import RotatingFileHandler
from dotenv import dotenv_values
from typing_extensions import Literal
from pathlib import Path
from datetime import datetime
from antispoofing.src.lit_models import BaseLitModel
import os
# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)
config = dotenv_values(".env")
DATA_DIR = Path(__file__).resolve().parents[1]


def _import_class(module_and_clas_name: str) -> type:
    """Import class from a module, e.g. 'antispoofing.src.data'"""
    module_name, class_name = module_and_clas_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument("--data_class", type=str, default="CASIA")
    parser.add_argument("--model_class", type=str, default="CNN")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(
        f"antispoofing.src.data.{temp_args.data_class}")
    model_class = _import_class(
        f"antispoofing.src.models.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    # lit_model_group = parser.add_argument_group("LitModel Args")
    # lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    # args = parser.parse_args()
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=CNN --data_class=CASIA

    python training/run_experiment.py

    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(
        f"antispoofing.src.data.{args.data_class}")
    model_class = _import_class(
        f"antispoofing.src.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)
    lit_model = BaseLitModel(model=model)

    # if args.loss not in ("ctc", "transformer"):
    #     lit_model_class = lit_models.BaseLitModel

    # if args.load_checkpoint is not None:
    #     lit_model = lit_model_class.load_from_checkpoint(
    #         args.load_checkpoint, args=args, model=model)
    # else:
    #     lit_model = lit_model_class(args=args, model=model)

    # logger = pl.loggers.TensorBoardLogger("training/logs")

    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor="val_loss", mode="min", patience=10)
    # model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    # )

    lit_model = BaseLitModel(model=model)
    # if args.loss not in ("ctc", "transformer"):
    #     lit_model_class = lit_models.BaseLitModel
    # # Hide lines below until Lab 3
    # if args.loss == "ctc":
    #     lit_model_class = lit_models.CTCLitModel
    # # Hide lines above until Lab 3

    # if args.load_checkpoint is not None:
    #     lit_model = lit_model_class.load_from_checkpoint(
    #         args.load_checkpoint, args=args, model=model)
    # else:
    #     lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )

    # bar = pl.callbacks.TQDMProgressBar(refresh_rate=12)

    # bar = LitProgressBar()
    # callbacks = [bar, early_stopping_callback, model_checkpoint_callback]
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # Print full summary of the model

    # por defecto logro realizar un numero de epocas = 162 para luego cargar test y obtener
    # Test metric             DataLoader 0
    # test_acc            0.6372548937797546

    # trainer = pl.Trainer.from_argparse_args(
    #     args, gpus=1, max_epochs=10, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    # trainer = pl.Trainer.from_argparse_args(
    #      args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    trainer = Trainer(gpus=1, precision=16, progress_bar_refresh_rate=5,
                      max_epochs=3, callbacks=callbacks)
    #trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    # trainer.test(lit_model, datamodule=data)

    CONFIG_DIR = Path(__file__).resolve().parents[1]
    print("Current Position: ", os.getcwd())
    best_model_path = model_checkpoint_callback.best_model_path
    print(best_model_path)
    print("config path: ", CONFIG_DIR)

    torch.save(lit_model, f"{CONFIG_DIR}/model.pt")

    return


if __name__ == '__main__':

    main()
