"""Implementation of detectors registration using the Fourier Phase loss"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale

from .interface import SAiryscanRegistration


class SRegisterFourierPhase(SAiryscanRegistration):
    """Register the detectors stack by translating each image to the detector position is array

    :param weight: Weight applied on the detector position translation

    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the registration

        :param image: Single channel/frame stack to register,
        :return: The registered stack
        """
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
    def hessian(img: torch.Tensor) -> torch.Tensor:
        """Sparse Hessian regularization term

        :param img: Tensor of shape BCYX containing the estimated image
        :return: THe hessian term
        """
        dxx2 = torch.square(-img[2:, 1:-1] + 2 * img[1:-1, 1:-1] - img[:-2, 1:-1])
        dyy2 = torch.square(-img[1:-1, 2:] + 2 * img[1:-1, 1:-1] - img[1:-1, :-2])
        dxy2 = torch.square(img[2:, 2:] - img[2:, 1:-1] - img[1:-1, 2:] +
                            img[1:-1, 1:-1])
        return dxx2 + dyy2 + 2 * dxy2

    @staticmethod
    def gradient(img: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the image

        :param img: Image to process,
        :return: THe gradient image of img
        """
        tv_h = torch.abs(img[1:, :] - img[:-1, :])
        tv_w = torch.abs(img[:, 1:] - img[:, :-1])
        return F.pad(tv_h, (0, 0, 1, 0), "constant", 0) + F.pad(tv_w, (1, 0, 0, 0), "constant", 0)

    @staticmethod
    def _translate_detector(img: torch.Tensor, translate: tuple[float, float]) -> torch.Tensor:
        """Translate one detector

        :param img: Tensor for one single detector
        :param translate: (x, y, z) or (x, y) translation vector
        :returns: the translated tensor
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
