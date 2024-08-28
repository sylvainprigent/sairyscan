"""Test the position registering implementation"""
import os
import numpy as np
from skimage.io import imread

from sairyscan.data import celegans
from sairyscan.registration import SRegisterPosition


def test_position_celegans():
    """Test the 2D position registration on the C. elegans image."""

    image = celegans()

    reg = SRegisterPosition()
    image_out = reg(image)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'celegans_position.tif')
    image_ref = imread(ref_file)

    np.testing.assert_almost_equal(image_out.detach().numpy(), image_ref, decimal=1)
