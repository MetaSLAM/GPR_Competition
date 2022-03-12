import numpy as np

from .geometry import lidar_trans, to_image
from .feature import HogFeature


def save_feature_for_submission(
    npy_name: str,
    feature: np.ndarray,
):
    """You can save the features of the testing/query set for
    submission. Note that we used L2-norm to search the nearest
    neighbors, so if you use cosine distance, remember to normalize
    you feature before calling this function.

    Args:
        npy_name: the name of *.npy file you want to save
        feature: should be (N_submap * feature_dim) np.ndarray
    """
    if feature.ndim != 2:
        raise ValueError('feature can only have size (N_submap * feature_dim)')

    np.save(npy_name, feature)
