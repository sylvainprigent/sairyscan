import os
import numpy as np
from skimage.io import imread, imsave

from sairyscan.reconstruction.pseudo_confocal import PseudoConfocal
from sairyscan.data import celegans


# tmp_path is a pytest fixture
def test_pseudo_confocal_2d(tmp_path):
    """An example of how you might test your plugin."""

    image = celegans()

    filter_ = PseudoConfocal(pinhole=1)
    out_image = filter_(image)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    #imsave(os.path.join(root_dir, 'celegans_pseudoconfocal.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_pseudoconfocal.tif'))

    np.testing.assert_equal(out_image.detach().numpy(), ref_image)
