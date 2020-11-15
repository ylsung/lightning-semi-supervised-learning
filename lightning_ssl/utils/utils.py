import os
import logging


# def create_logger(save_path="", file_type="", level="debug"):

#     if level == "debug":
#         _level = logging.DEBUG
#     elif level == "info":
#         _level = logging.INFO

#     logger = logging.getLogger()
#     logger.setLevel(_level)

#     cs = logging.StreamHandler()
#     cs.setLevel(_level)
#     logger.addHandler(cs)

#     if save_path != "":
#         file_name = os.path.join(save_path, file_type + "_log.txt")
#         fh = logging.FileHandler(file_name, mode="w")
#         fh.setLevel(_level)

#         logger.addHandler(fh)

#     return logger


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def count_parameters(model):
    # copy from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    # baldassarre.fe's reply
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
