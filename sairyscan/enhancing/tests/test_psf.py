import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sairyscan.enhancing._psfs import PSFGaussian


# tmp_path is a pytest fixture
def test_gaussian_2d(tmp_path):
    """An example of how you might test your plugin."""

    psf_generator = PSFGaussian((1.5, 1.5), (15, 15))
    psf = psf_generator()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    # imsave(os.path.join(root_dir, 'psf.tif'), psf.detach().numpy())
    ref_psf = imread(os.path.join(root_dir, 'psf.tif'))

    np.testing.assert_equal(psf.detach().numpy(), ref_psf)
