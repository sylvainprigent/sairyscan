import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sairyscan.enhancing.gaussian import SAiryscanGaussian


# tmp_path is a pytest fixture
def test_gaussian_filter_2d(tmp_path):
    """An example of how you might test your plugin."""

    root_dir = os.path.dirname(os.path.abspath(__file__))
    my_test_file = os.path.join(root_dir, 'celegans_ism.tif')

    image = torch.Tensor(np.float32(imread(my_test_file)))

    filter_ = SAiryscanGaussian(sigma=0.5, kernel_size=7)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_ism_gaussian.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_ism_gaussian.tif'))

    np.testing.assert_almost_equal(out_image.detach().numpy(), ref_image, decimal=2)
