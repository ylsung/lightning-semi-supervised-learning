import os
import torch
import numpy as np
from lightning_ssl.module import ClassifierModule, MixmatchModule
from lightning_ssl.dataloader import SemiCIFAR10Module, SupervisedCIFAR10Module
from lightning_ssl.models import WideResNet
import pytorch_lightning as pl
from lightning_ssl.utils import count_parameters
from lightning_ssl.utils.argparser import parser, print_args


if __name__ == "__main__":
    args = parser()

    gpus = args.gpus if torch.cuda.is_available() else None

    # set the random seed
    # np.random.seed(args.seed) # for sampling labeled/unlabeled/val dataset
    pl.seed_everything(args.seed)

    # load the data and classifier
    if args.dataset == "cifar10":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedCIFAR10Module

        elif args.learning_scenario == "semi":
            loader_class = SemiCIFAR10Module

        data_loader = loader_class(
            args,
            args.data_root,
            args.num_workers,
            args.batch_size,
            args.num_labeled,
            args.num_val,
            args.num_augments,
        )
        classifier = WideResNet(depth=28, num_classes=data_loader.n_classes)

    else:
        raise NotImplementedError

    if args.learning_scenario == "supervised":
        module = ClassifierModule
    else:  # semi supervised learning algorithm
        if args.algo == "mixmatch":
            module = MixmatchModule
        else:
            raise NotImplementedError

    print(f"model paramters: {count_parameters(classifier)} M")

    # set the number of classes in the args
    setattr(args, "n_classes", data_loader.n_classes)

    data_loader.prepare_data()
    data_loader.setup()

    model = module(args, classifier, loaders=None)

    # trainer = Trainer(tr_loaders, va_loader, te_loader)

    if args.todo == "train":
        print(
            f"labeled size: {data_loader.num_labeled_data}"
            f"unlabeled size: {data_loader.num_unlabeled_data}, "
            f"val size: {data_loader.num_val_data}"
        )

        save_folder = (
            f"{args.dataset}_{args.learning_scenario}_{args.algo}_{args.affix}"
        )
        # model_folder = os.path.join(args.model_root, save_folder)
        # checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(filepath=model_folder)

        # tt_logger = pl.loggers.TestTubeLogger("tt_logs", name=save_folder, create_git_tag=True)
        # tt_logger.log_hyperparams(args)

        # if True:
        #     print_args(args)

        #     trainer = pl.trainer.Trainer(gpus=gpus, max_epochs=args.max_epochs)
        # else:

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

        ckpt = pl.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_path, "last"))

        setattr(args, "checkpoint_folder", ckpt_path)

        print_args(args)

        trainer = pl.trainer.Trainer(
            gpus=gpus,
            max_steps=args.max_steps,
            logger=tb_logger,
            max_epochs=args.max_epochs,
            checkpoint_callback=ckpt,
            benchmark=True,
            profiler=True,
            progress_bar_refresh_rate=args.progress_bar_refresh_rate,
            reload_dataloaders_every_epoch=True,
        )

        trainer.fit(model, datamodule=data_loader)
        trainer.test()
    else:
        trainer = pl.trainer.Trainer(resume_from_checkpoint=args.load_checkpoint)
        trainer.test(model, datamodule=data_loader)
