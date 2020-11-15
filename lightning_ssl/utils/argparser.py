import yaml
import argparse
from configparser import ConfigParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parser():
    conf_parser = argparse.ArgumentParser(
        description="parser for config",
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False,
    )

    conf_parser.add_argument(
        "-c", "--conf_file", help="Specify config file", metavar="FILE"
    )

    conf_parser.add_argument(
        "-d", "--dataset", default="cifar10", help="use what dataset"
    )

    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}
    if args.conf_file:
        with open(args.conf_file, "r") as f:
            configs = yaml.load(f, yaml.SafeLoader)
            assert args.dataset in configs, f"Don't have the config for {args.dataset}"
            defaults.update(configs[args.dataset])

    parser = argparse.ArgumentParser(
        description="Semi-Supervised Learning", parents=[conf_parser]
    )

    parser.add_argument(
        "--learning_scenario",
        default="semi",
        choices=["semi", "supervised"],
        help="what learning scenario to use",
    )

    parser.add_argument("--algo", default="none", help="which algorithm to use")

    parser.add_argument(
        "--todo",
        choices=["train", "test"],
        default="train",
        help="what behavior want to do: train | test",
    )

    parser.add_argument(
        "--data_root", default=".", help="the directory to save the dataset"
    )

    parser.add_argument(
        "--log_root",
        default=".",
        help="the directory to save the logs or other imformations (e.g. images)",
    )

    parser.add_argument(
        "--model_root", default="checkpoint", help="the directory to save the models"
    )

    parser.add_argument("--load_checkpoint", default="./model/default/model.pth")

    parser.add_argument(
        "--affix", default="default", help="the affix for the save folder"
    )

    parser.add_argument("--seed", type=int, default=1, help="seed")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="how many workers used in data loader",
    )

    parser.add_argument("--batch_size", "-b", type=int, default=128, help="batch size")

    parser.add_argument(
        "--num_labeled", type=int, default=1000, help="number of labeled data"
    )

    parser.add_argument(
        "--num_val", type=int, default=5000, help="the amount of validation data"
    )

    parser.add_argument(
        "--max_epochs",
        "-m_e",
        type=int,
        default=200,
        help="the maximum numbers of the model see a sample",
    )

    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-2, help="learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        "-w",
        type=float,
        default=2e-4,
        help="the parameter of l2 restriction for weights",
    )

    parser.add_argument("--gpus", "-g", default="0", help="what gpus to use")

    parser.add_argument(
        "--n_eval_step",
        type=int,
        default=100,
        help="number of iteration per one evaluation",
    )

    parser.add_argument(
        "--n_checkpoint_step",
        type=int,
        default=4000,
        help="number of iteration to save a checkpoint",
    )

    parser.add_argument(
        "--n_store_image_step",
        type=int,
        default=4000,
        help="number of iteration to save adversaries",
    )

    parser.add_argument(
        "--max_steps", type=int, default=1 << 16, help="maximum iteration for training"
    )

    parser.add_argument(
        "--ema",
        type=float,
        default=0.999,
        help="the decay for exponential moving average",
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="the paramters for label smoothing",
    )

    parser.add_argument(
        "--augment", action="store_true", help="whether to augment the training data"
    )

    parser.add_argument(
        "--num_augments",
        type=int,
        default=2,
        help="how many augment samples for unlabeled data in Mixmatch",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=-1,
        help="the hyperparameter for beta distribution in mixup \
            (0 < alpha. If alpha < 0 means no mix)",
    )

    parser.add_argument(
        "--T",
        type=float,
        default=0.5,
        help="temperature for softmax distribution or sharpen parameter in mixmatch",
    )

    parser.add_argument(
        "--lambda_u",
        type=float,
        default=100,
        help="the weight of the loss for the unlabeled data",
    )

    parser.add_argument(
        "--batch_inference",
        type=str2bool,
        default=False,
        help="whether use batch inference in generating psuedo labels and computing loss",
    )

    parser.add_argument(
        "--progress_bar_refresh_rate",
        type=int,
        default=1,
        help="The frequency to refresh the progress bar (0 diables the progress bar)",
    )

    parser.set_defaults(**defaults)
    parser.set_defaults(**vars(args))

    return parser.parse_args(remaining_argv)


def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info("{:<16} : {}".format(k, v))
        else:
            print("{:<16} : {}".format(k, v))
