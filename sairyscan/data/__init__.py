"""This module gives access to example data for testing and demo SAiryscan"""
import os.path as osp
import os
from sairyscan.core import SAiryscanReader

__all__ = ['celegans']

legacy_data_dir = osp.abspath(osp.dirname(__file__))


def _fetch(data_filename):
    """Fetch a given data file from the local data dir.
    This function provides the path location of the data file given
    its name in the scikit-image repository.
    Parameters
    ----------
    data_filename:
        Name of the file in the scikit-bioimaging data dir
    Returns
    -------
    Path of the local file as a python string.
    """

    filepath = os.path.join(legacy_data_dir, data_filename)

    if os.path.isfile(filepath):
        return filepath
    raise FileExistsError("Cannot find the file:", filepath)


def _load(f):
    """Load an image file located in the data directory.
    Parameters
    ----------
    f : string
        File name.
    Returns
    -------
    img : ndarray
        Image loaded from ``skimage.data_dir``.
    """
    # importing io is quite slow since it scans all the backends
    # we lazy import it here
    from skimage.io import imread
    return imread(_fetch(f))


def celegans():
    """2D C. elegans sample expressing the ERM-1::mNeonGreen fusion protein.

    Returns
    -------
    celegans : (32, 316, 316) float ndarray
    """

    reader = SAiryscanReader(_fetch("celegans_airyscan.czi"))
    return reader.data()
