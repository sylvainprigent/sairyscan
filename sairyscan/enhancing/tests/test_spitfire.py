import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sairyscan.enhancing.spitfire_deconv import SpitfireDeconv
from sairyscan.enhancing.spitfire_denoise import SpitfireDenoise
from sairyscan.enhancing._psfs import PSFGaussian


# tmp_path is a pytest fixture
def test_spitfire_deconv_2d(tmp_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    my_test_file = os.path.join(root_dir, 'celegans_ism.tif')

    image = torch.Tensor(np.float32(imread(my_test_file)))

    psf_generator = PSFGaussian((1.5, 1.5), (15, 15))
    psf = psf_generator()

    filter_ = SpitfireDeconv(psf, weight=0.6, reg=0.995)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_ism_spitfire_deconv_2d.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_ism_spitfire_deconv_2d.tif'))

    np.testing.assert_almost_equal(out_image.detach().numpy(), ref_image, decimal=3)


def test_spitfire_denoise_2d(tmp_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    my_test_file = os.path.join(root_dir, 'celegans_ism.tif')

    image = torch.Tensor(np.float32(imread(my_test_file)))

    filter_ = SpitfireDenoise(weight=0.6, reg=0.85)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_ism_spitfire_denoise_2d.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_ism_spitfire_denoise_2d.tif'))

    np.testing.assert_almost_equal(out_image.detach().numpy(), ref_image, decimal=1)
