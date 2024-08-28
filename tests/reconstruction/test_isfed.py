import os
import numpy as np
from skimage.io import imread, imsave

from sairyscan.reconstruction.isfed import ISFED
from sairyscan.registration.position import SRegisterPosition
from sairyscan.data import celegans


# tmp_path is a pytest fixture
def test_ifed_2d(tmp_path):
    """An example of how you might test your plugin."""

    image = celegans()

    reg = SRegisterPosition()
    reg_image = reg(image)

    filter_ = ISFED(epsilon=0.9)
    out_image = filter_(image, reg_image)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    #imsave(os.path.join(root_dir, 'celegans_isfed.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_isfed.tif'))

    np.testing.assert_almost_equal(out_image.detach().numpy(), ref_image, decimal=1)
