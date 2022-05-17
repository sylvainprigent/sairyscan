import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from .interface import SairyscanRegistration
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage.transform import rescale


class SRegisterFourierPhase(SairyscanRegistration):
    """Register the detectors stack by translating each image to the detector position is array

    Parameters
    ----------
    weight: int
        Weight applied on the detector position translation

    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def __call__(self, image):
        # image_np = image.detach().numpy()
        image_out = torch.zeros(image.shape, dtype=torch.float32)
        image_out[0, ...] = image[0, ...].clone()
        ratio = (image.shape[1]-5)/image.shape[1]
        for i in range(1, 32):
            shift, _, _ = phase_cross_correlation(
                                            rescale(image[0, ...].detach().numpy(), 2),
                                            rescale(image[i, ...].detach().numpy(), 2),
                                            normalization=None,
                                            overlap_ratio=ratio
                                            )
            print(f'shift {i}= [{shift[1] / 2 }, {shift[0] / 2 }]')
            image_out[i, ...] = self._translate_detector(image[i, ...], (shift[1]/2, shift[0]/2))
        return image_out

    @staticmethod
    def hessian(img):
        """Sparse Hessian regularization term

        Parameters
        ----------
        img: Tensor
            Tensor of shape BCYX containing the estimated image

        """
        dxx2 = torch.square(-img[2:, 1:-1] + 2 * img[1:-1, 1:-1] - img[:-2, 1:-1])
        dyy2 = torch.square(-img[1:-1, 2:] + 2 * img[1:-1, 1:-1] - img[1:-1, :-2])
        dxy2 = torch.square(img[2:, 2:] - img[2:, 1:-1] - img[1:-1, 2:] +
                            img[1:-1, 1:-1])
        return dxx2 + dyy2 + 2 * dxy2

    @staticmethod
    def gradient(img):
        tv_h = torch.abs(img[1:, :] - img[:-1, :])
        tv_w = torch.abs(img[:, 1:] - img[:, :-1])
        return F.pad(tv_h, (0, 0, 1, 0), "constant", 0) + F.pad(tv_w, (1, 0, 0, 0), "constant", 0)

    @staticmethod
    def _translate_detector(img, translate):
        """Translate one detector

        Parameters
        ----------
        img: Tensor
            Tensor for one single detector
        translate: tuple
            (x, y, z) or (x, y) translation vector

        Returns
        -------
        Tensor: the translated tensor
        """
        img_bc = img.view((1, 1, img.shape[0], img.shape[1]))
        img_out = transforms.functional.affine(img_bc, angle=0, translate=translate, scale=1,
                                               shear=[0, 0],
                                               interpolation=transforms.functional.InterpolationMode.BILINEAR)
        return img_out.view(img.shape)


metadata = {
    'name': 'SRegisterFourierPhase',
    'class': SRegisterFourierPhase,
    'parameters': {
    }
}
