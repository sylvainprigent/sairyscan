import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sairyscan.reconstruction.spitfire_join_deconv import SpitfireJoinDeconv
from sairyscan.registration.position import SRegisterPosition
from sairyscan.enhancing._psfs import PSFGaussian
from sairyscan.data import celegans


# tmp_path is a pytest fixture
def test_spitfire_join_deconv_2d(tmp_path):
    """An example of how you might test your plugin."""

    image = celegans()

    psf_generator = PSFGaussian((1.5, 1.5), (15, 15))
    psf = psf_generator()

    reg = SRegisterPosition()
    reg_image = reg(image)

    filter_ = SpitfireJoinDeconv(psf, weight=0.6, reg=0.995, detector_weights='mean')
    out_image = filter_(reg_image)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    # imsave(os.path.join(root_dir, 'celegans_spitfire_join_deconv.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_spitfire_join_deconv.tif'))

    np.testing.assert_almost_equal(out_image.detach().numpy(), ref_image, decimal=5)
