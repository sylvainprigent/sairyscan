"""This module gives access to example data for testing and demo SAiryscan"""
import os.path as osp
import os
import numpy as np
import torch

from sairyscan.core import SAiryscanReader

__all__ = ['celegans']

legacy_data_dir = osp.abspath(osp.dirname(__file__))


def _fetch(data_filename: str):
    """Fetch a given data file from the local data dir.
    This function provides the path location of the data file given
    its name in the scikit-image repository.

    :param data_filename: Name of the file in the library data dir,
    :return: Path of the local file as a python string.
    """

    filepath = os.path.join(legacy_data_dir, data_filename)

    if os.path.isfile(filepath):
        return filepath
    raise FileExistsError("Cannot find the file:", filepath)


def _load(f: str) -> np.ndarray:
    """Load an image file located in the data directory.

    :param f : File name.
    :return img : Image loaded from ``skimage.data_dir``.
    """
    # importing io is quite slow since it scans all the backends
    # we lazy import it here
    from skimage.io import imread
    return imread(_fetch(f))


def celegans() -> torch.Tensor:
    """2D Celegans sample expressing the ERM-1::mNeonGreen fusion protein.

    :return: celegans : (32, 316, 316) float ndarray
    """

    reader = SAiryscanReader(_fetch("celegans_airyscan.czi"))
    return reader.data()
