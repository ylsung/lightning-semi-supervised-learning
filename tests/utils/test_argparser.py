import os
import sys
import copy
import pathlib
from lightning_ssl.utils.argparser import parser

PATH = pathlib.Path(__file__).parent
CONFIG_PATH = os.path.join(PATH, "assets", "tmp_config.yml")


def test_parser():
    original_argv = copy.deepcopy(sys.argv)
    sys.argv = [
        "temp.py",
        "--c",
        f"{CONFIG_PATH}",
        "--learning_scenario",
        "supervised",
        "--todo",
        "train",
    ]

    args = parser()
    sys.argv = original_argv

    assert args.learning_scenario == "supervised"
    assert args.todo == "train"
    assert args.affix == "test"
