import os
import torch
import numpy as np
from argparser import parser, print_args
from module import ClassifierModule, MixmatchModule
from dataloader import SemiCIFAR10Loader, SupervisedCIFAR10Loader
from models import WideResNet
import pytorch_lightning as pl
from utils import count_parameters

# class Trainer:
#     def __init__(self, tr_loaders, va_loader, te_loader):
#         self.tr_loaders = tr_loaders
#         self.va_loaders = va_loaders
#         self.te_loaders = te_loaders
#         pass
#     def fit(self, module):
#         pass

if __name__ == '__main__':
    args = parser()

    # to avoid using the gpu 0 when using single gpu, seems like bug in pytorch
    # if len(args.gpus) == 1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # gpus = 1 if len(args.gpus) == 1 else args.gpus

    gpus = args.gpus

    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    # print(list(range(torch.cuda.device_count())))

    # exit()

    # set the random seed
    np.random.seed(args.seed) # for sampling labeled/unlabeled/val dataset
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # load the data and classifier
    if args.dataset == "cifar10":
        loader_class = eval(args.learning_scenario.capitalize() + args.dataset.upper() + "Loader")
        data_loader = loader_class(args, args.data_root, args.num_workers, args.batch_size, 
            args.num_labeled, args.num_valid, args.num_augments)
        classifier = WideResNet(depth=28, num_classes=data_loader.n_classes)

    print("model paramters: %d M" % count_parameters(classifier))

    # set the number of classes in the args
    setattr(args, "n_classes", data_loader.n_classes)

    tr_loader = data_loader.get_train_loader()
    va_loader = data_loader.get_valid_loader()
    te_loader = data_loader.get_test_loader()
    loaders = {"tr_loader": tr_loader, "va_loader": va_loader, "te_loader": te_loader}

    if args.learning_scenario == "supervised":
        module = ClassifierModule
    else:
        module = eval(args.algo.capitalize() + "Module")

    model = module(args, classifier, loaders=loaders)
    
    # trainer = Trainer(tr_loaders, va_loader, te_loader)    

    if args.todo == "train":
        print("labeled size: %d, unlabeled size: %d, valid size: %d" % (
        data_loader.num_labeled_data, data_loader.num_unlabeled_data, data_loader.num_valid_data))

        save_folder = "%s_%s_%s_%s" % (args.dataset, args.learning_scenario, args.algo, args.affix)
        # model_folder = os.path.join(args.model_root, save_folder)
        # checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(filepath=model_folder)

        # tt_logger = pl.loggers.TestTubeLogger("tt_logs", name=save_folder, create_git_tag=True)
        # tt_logger.log_hyperparams(args)

        # if True:
        #     print_args(args)
            
        #     trainer = pl.trainer.Trainer(gpus=gpus, max_epochs=args.max_epochs)
        # else:

        tb_logger = pl.loggers.TensorBoardLogger("lightning_logs", name=save_folder)
        tb_logger.log_hyperparams(args)

        # set the path of checkpoint
        save_dir = (getattr(tb_logger, 'save_dir', None) or
                    getattr(tb_logger, '_save_dir', None))
        ckpt_path = os.path.join(
            save_dir,
            tb_logger.name,
            f'version_{tb_logger.version}',
            "checkpoints"
        )

        ckpt = pl.callbacks.ModelCheckpoint(
                            filepath=os.path.join(ckpt_path, "{epoch}-{val_acc:.2f}"),
                            monitor="val_acc",
                            mode="max")

        setattr(args, "checkpoint_folder", ckpt_path)
        
        print_args(args)
        
        trainer = pl.trainer.Trainer(gpus=gpus, max_steps=args.max_steps, 
            logger=tb_logger, max_epochs=args.max_epochs, checkpoint_callback=ckpt,
            benchmark=True, profiler=True, progress_bar_refresh_rate=1, 
            log_save_interval=100, row_log_interval=10)

        trainer.fit(model)
        trainer.test()
    else:
        trainer = pl.trainer.Trainer(resume_from_checkpoint=args.load_checkpoint)
        trainer.test(model)
