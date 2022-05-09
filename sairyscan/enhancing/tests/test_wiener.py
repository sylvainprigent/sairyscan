import os
import numpy as np
from skimage.io import imread

from sairyscan.wiener import SAiryscanWiener
from sairyscan.psfs import PSFGaussian


# tmp_path is a pytest fixture
def test_wiener_2d(tmp_path):
    """An example of how you might test your plugin."""

    root_dir = os.path.dirname(os.path.abspath(__file__))
    my_test_file = os.path.join(root_dir, 'ism_celegans.tif')

    image = torch.Tensor(imread(my_test_file))

    psf_generator = PSFGaussian((1.5, 1.5), (316, 316))
    psf = psf_generator()

    filter_ = SAiryscanWiener(psf, beta=0.001)
    out_image = filter_(image)

    np.testing.assert_equal(out_image.to_numpy(), out_image.to_numpy())