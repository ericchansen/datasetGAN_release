import argparse
import logging
import logging.config
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats
from torch.cuda import random

STYLEGAN_DIR = "../models"
sys.path.append(STYLEGAN_DIR)
from stylegan1 import G_mapping, Truncation

logger = logging.getLogger(__name__)
with open("logging.yml", "r") as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)


def main(np_binary_a_path, np_binary_b_path):
    try:
        array_a = np.load(np_binary_a_path)
    except ValueError:
        array_a = []
        with open(np_binary_a_path ,"rb") as f:
            array_a.append(np.load(f))
    array_b = np.load(np_binary_b_path)

    # stats_a = stats.describe(array_a)
    # stats_b = stats.describe(array_b)
    # logger.info(stats.describe(np_binary_a))
    # logger.info(stats.describe(np_binary_b))

    euclidean_distance = l2_norm = np.linalg.norm(array_a - array_b)
    logger.info(f"euclidean distance: {euclidean_distance}")

    all_close = np.allclose(array_a, array_b)
    logger.info(f"numpy.allclose: {all_close}")

    return euclidean_distance


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"torch device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("np_binary_a_path", type=Path)
    parser.add_argument("np_binary_b_path", type=Path)
    opts = parser.parse_args()

    main(
        opts.np_binary_a_path, opts.np_binary_b_path,
    )
