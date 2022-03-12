"""This script provides functions for grading. 
It is aimed for the organizers.

Author: Haowen Lai
"""

import numpy as np


def gen_random_sequence(total_num: int, seed: int) -> np.ndarray:
    """This random sequence is used to shuffle our submaps.
    Args:
        total_num: total number of submaps. Also the length of seq.
        seed: the seed to initialize RNG
    Returns:
        seq: the sequence, size (total_num, )
    """
    np.random.seed(seed=seed)
    return np.random.permutation(total_num)
