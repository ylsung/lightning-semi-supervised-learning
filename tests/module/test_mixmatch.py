import os
import sys
import copy
import torch
import pytest
import pathlib
import numpy as np
from tests.module.simple_model import SimpleModel
from lightning_ssl.module import ClassifierModule, MixmatchModule
from lightning_ssl.dataloader import (
    SemiCIFAR10Module,
    SupervisedCIFAR10Module,
)
from lightning_ssl.models import WideResNet
import pytorch_lightning as pl
from lightning_ssl.utils import count_parameters
from lightning_ssl.utils.argparser import parser, print_args

PATH = pathlib.Path(__file__).parent
CONFIG_PATH = os.path.join(PATH, "assets", "tmp_config.yml")


@pytest.mark.parametrize("batch_inference", ["True", "False"])
def test_mixmatch(tmpdir, batch_inference):
    original_argv = copy.deepcopy(sys.argv)
    sys.argv = [
        "tmp.py",
        "-c",
        f"{CONFIG_PATH}",
        "--log_root",
        str(tmpdir),
        "--batch_inference",
        batch_inference,
    ]
    print(sys.argv)
    args = parser()
    sys.argv = original_argv

    gpus = args.gpus if torch.cuda.is_available() else None

    # load the data and classifier

    data_loader = SemiCIFAR10Module(
        args,
        args.data_root,
        args.num_workers,
        args.batch_size,
        args.num_labeled,
        args.num_val,
        args.num_augments,
    )
    classifier = WideResNet(depth=10, num_classes=data_loader.n_classes)

    print(f"model paramters: {count_parameters(classifier)} M")

    # set the number of classes in the args
    setattr(args, "n_classes", data_loader.n_classes)

    data_loader.prepare_data()
    data_loader.setup()

    model = MixmatchModule(args, classifier, loaders=None)

    print(
        f"labeled size: {data_loader.num_labeled_data}"
        f"unlabeled size: {data_loader.num_unlabeled_data}, "
        f"val size: {data_loader.num_val_data}, "
        f"test size: {data_loader.num_test_data}"
    )

    save_folder = f"{args.dataset}_{args.learning_scenario}_{args.algo}_{args.affix}"

    tb_logger = pl.loggers.TensorBoardLogger(
        os.path.join(args.log_root, "lightning_logs"), name=save_folder
    )
    tb_logger.log_hyperparams(args)

    # set the path of checkpoint
    save_dir = getattr(tb_logger, "save_dir", None) or getattr(
        tb_logger, "_save_dir", None
    )
    ckpt_path = os.path.join(
        save_dir, tb_logger.name, f"version_{tb_logger.version}", "checkpoints"
    )

    ckpt = pl.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_path, "{epoch}"))

    setattr(args, "checkpoint_folder", ckpt_path)

    print_args(args)

    trainer = pl.trainer.Trainer(
        gpus=gpus,
        logger=tb_logger,
        checkpoint_callback=ckpt,
        fast_dev_run=True,
        reload_dataloaders_every_epoch=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
    )

    trainer.fit(model, datamodule=data_loader)
    trainer.test()
