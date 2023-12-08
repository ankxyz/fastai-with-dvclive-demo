import json
import logging
from pathlib import Path

from dvclive.fastai import DVCLiveCallback
from fastai.vision.augment import aug_transforms
from fastai.vision.core import defaults
from fastai.vision.data import ImageDataLoaders, Normalize, imagenet_stats
from fastai.vision.learner import vision_learner, models
from fastai.metrics import error_rate, accuracy
from fastai.callback.all import *
from fastai.vision.augment import Resize
import torch


logging.basicConfig(level=logging.INFO, format="TRAIN: %(asctime)s: %(message)s")


BATCH_SIZE = 8
IMAGE_SIZE = 224
EPOCHS = 3
LR_MAX = slice(1e-5,1e-4)


if __name__ == "__main__":

    logging.info("Set default compute device")
    defaults.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device = {defaults.device}")

    logging.info("Prepare transformers")
    tfms = [aug_transforms(), Normalize.from_stats(*imagenet_stats)]

    logging.info("Create data loader")
    data = ImageDataLoaders.from_folder(
        path="./data",
        item_tfms=Resize(IMAGE_SIZE),
        ds_tfms=tfms,
        size=IMAGE_SIZE,
        bs=BATCH_SIZE
    )

    logging.info("Create and fit learner")
    required_metrics = [error_rate, accuracy]
    learn = vision_learner(data, models.resnet34, metrics=required_metrics)
    learn.unfreeze()
    dvc_cb = DVCLiveCallback(dvcyaml="dvclive/dvc.yaml",  save_dvc_exp=False)
    learn.fit_one_cycle(EPOCHS, lr_max=LR_MAX, cbs=[dvc_cb])

    logging.info("Save learner")
    dataset_size = len(data.dataset.items)
    model_path = Path("./models").absolute() / "model.pkl"
    learn.export(fname=model_path)
    logging.info(f"Model (learner) saved in {model_path}")

    logging.info("Get train metrics")
    train_metrics_list = learn.recorder._train_mets
    train_metrics = {
        metric.name: metric.value.item()
        for metric in train_metrics_list
    }

    logging.info("Build train report")
    train_report = {
        "dataset_size": dataset_size,
        "train_metrics": train_metrics
    }

    logging.info("Save train report")
    report_path = Path("./reports") / "train_report.json"

    with open(report_path, "w") as train_report_f:
        json.dump(obj=train_report, fp=train_report_f, indent=4)
