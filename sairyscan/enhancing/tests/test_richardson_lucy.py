import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sairyscan.enhancing.richardson_lucy import SAiryscanRichardsonLucy
from sairyscan.enhancing._psfs import PSFGaussian


# tmp_path is a pytest fixture
def test_richardson_lucy_2d(tmp_path):
    """An example of how you might test your plugin."""

    root_dir = os.path.dirname(os.path.abspath(__file__))
    my_test_file = os.path.join(root_dir, 'celegans_ism.tif')

    image = torch.Tensor(np.float32(imread(my_test_file)))

    psf_generator = PSFGaussian((1.5, 1.5), (15, 15))
    psf = psf_generator()

    filter_ = SAiryscanRichardsonLucy(psf, niter=30)
    out_image = filter_(image)

    #imsave(os.path.join(root_dir, 'celegans_ism_richardson_lucy.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_ism_richardson_lucy.tif'))

    np.testing.assert_equal(out_image.detach().numpy(), ref_image)
