import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu


class SureMap:
    """Calculate the Sure weights map for IFED and ISFED

    Parameters
    ----------
    smooth: bool
        True to smooth the map with a sigma=2 gaussian filter
    """
    def __init__(self, smooth=False):
        self.patch_size = 5
        self.smooth = smooth

    def __call__(self, image_ref, image_a, image_b):
        width = image_ref.shape[0]
        height = image_ref.shape[1]
        image_ref = image_ref.view(1, 1, image_ref.shape[0], image_ref.shape[1])
        image_a = image_a.view(1, 1, image_a.shape[0], image_a.shape[1])
        image_b = image_b.view(1, 1, image_b.shape[0], image_b.shape[1])

        # extract patch
        image_ref_patch = torch.nn.functional.unfold(image_ref, self.patch_size,
                                                     padding=self.patch_size//2)
        image_a_patch = torch.nn.functional.unfold(image_a, self.patch_size,
                                                   padding=self.patch_size//2)
        image_b_patch = torch.nn.functional.unfold(image_b, self.patch_size,
                                                   padding=self.patch_size//2)

        # sure
        num = torch.sum((image_ref_patch-image_a_patch)*image_b_patch, dim=1)
        den = torch.sum(image_b_patch*image_b_patch, dim=1)
        sure_map = -num/den
        sure_map = sure_map.view(width, height)

        if self.smooth:
            return transforms.functional.gaussian_blur(sure_map.view(1, 1, width, height),
                                                       kernel_size=(7, 7),
                                                       sigma=2).view(width, height)
        else:
            return sure_map

    def mode(self, image_ref, image_a, image_b):
        sure_map = self.__call__(image_ref, image_a, image_b).detach().numpy()
        th = threshold_otsu(sure_map)
        return np.mean(sure_map[sure_map >= th])
