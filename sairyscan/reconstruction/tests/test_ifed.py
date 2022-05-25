import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sairyscan.reconstruction.ifed import IFED
from sairyscan.data import celegans


# tmp_path is a pytest fixture
def test_ifed_2d(tmp_path):
    """An example of how you might test your plugin."""

    image = celegans()

    filter_ = IFED(inner_ring_index=7, epsilon=0.3)
    out_image = filter_(image)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    # imsave(os.path.join(root_dir, 'celegans_ifed.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_ifed.tif'))

    np.testing.assert_equal(out_image.detach().numpy(), ref_image)
