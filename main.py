import os
import torch
import numpy as np
from argparser import parser, print_args
from ssl_module import ClassifierModule
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
    print_args(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load the data and classifier
    if args.dataset == "cifar10":
        loader_class = eval(args.learning_scenario.capitalize() + args.dataset.upper() + "Loader")
        data_loader = loader_class(args, args.data_root, args.num_workers, args.batch_size, 
            args.num_labeled, args.valid_percent)
        classifier = WideResNet(depth=28, num_classes=data_loader.n_classes)

    print("model paramters: %.4f M" % (count_parameters(classifier) / 1e6))

    # set the number of classes in the args
    setattr(args, "n_classes", data_loader.n_classes)

    tr_loader = data_loader.get_train_loader()
    va_loader = data_loader.get_valid_loader()
    te_loader = data_loader.get_test_loader()
    loaders = {"tr_loader": tr_loader, "va_loader": va_loader, "te_loader": te_loader}
    model = ClassifierModule(args, classifier, loaders=loaders)
    
    # trainer = Trainer(tr_loaders, va_loader, te_loader)    

    if args.todo == "train":
        print("labeled size: %d, unlabeled size: %d, valid size: %d" % (
        data_loader.num_labeled_data, data_loader.num_unlabeled_data, data_loader.num_valid_data))

        save_folder = "%s_%s_%s" % (args.dataset, args.learning_scenario, args.affix)
        model_folder = os.path.join(args.model_root, save_folder)
        # checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(filepath=model_folder)

        # tt_logger = pl.loggers.TestTubeLogger("tt_logs", name=save_folder, create_git_tag=True)
        # tt_logger.log_hyperparams(args)
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
        
        trainer = pl.trainer.Trainer(gpus=args.gpus, max_steps=args.max_steps, 
            logger=tb_logger, max_epochs=args.max_epochs, checkpoint_callback=ckpt)

        trainer.fit(model)
        trainer.test()
    else:
        trainer = pl.trainer.Trainer(resume_from_checkpoint=args.load_checkpoint)
        trainer.test(model)
