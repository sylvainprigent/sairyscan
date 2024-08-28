"""This module implement the SURE estimation of weights between two images"""
import numpy as np
import torch
from torchvision import transforms
from skimage.filters import threshold_otsu


class SureMap:
    """Calculate the Sure weights map for IFED and ISFED

    :param smooth: True to smooth the map with a sigma=2 gaussian filter
    """
    def __init__(self, smooth: float = False):
        self.patch_size = 5
        self.smooth = smooth

    def __call__(self,
                 image_ref: torch.Tensor,
                 image_a: torch.Tensor,
                 image_b: torch.Tensor
                 ) -> torch.Tensor:
        """Do the calculation

        :param image_ref: Reference image
        :param image_a: First image to combine
        :param image_b: Second image to combine
        :return: The sure map, where each coefficient the weight between the two images
        """
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
        return sure_map

    def mode(self,
             image_ref: torch.Tensor,
             image_a: torch.Tensor,
             image_b: torch.Tensor
             ) -> float:
        """Calculate a single weight coefficient to combine image_a and image_b

        :param image_ref: Reference image
        :param image_a: First image to combine
        :param image_b: Second image to combine
        :return: The weighting coefficient
        """
        sure_map = self(image_ref, image_a, image_b).detach().numpy()
        th = threshold_otsu(sure_map)
        return np.mean(sure_map[sure_map >= th])
