import logging
import os
import sys


def setup_logger(save_dir, filename="log.txt"):
    logger = logging.getLogger('logger')  # 必须每次都是一样，否则其他文件getLogger时候不是同一个对象
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger():
    return logging.getLogger('logger')
