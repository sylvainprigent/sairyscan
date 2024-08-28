"""Implementation of detector registration using the MSE loss"""
import numpy as np
import torch
from torchvision import transforms
from skimage.transform import rescale

from .interface import SAiryscanRegistration


class SRegisterMSE(SAiryscanRegistration):
    """Register the detectors stack by translating each image to the detector position is array

    :param weight: Weight applied on the detector position translation
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """DO the registration

        :param image: Single channel/frame stack to register,
        :return: The registered stack
        """
        self.progress(0)
        image_out = torch.zeros(image.shape, dtype=torch.float32)
        image_out[0, ...] = image[0, ...].clone()

        ref_image = rescale(image[0, ...].detach().numpy(), 2)
        # ref_image = ref_image/np.max(ref_image)
        for d in range(1, 32):
            self.progress(int(100*d/32))
            mse = np.zeros((13, 13))
            mov_image_ref = rescale(image[d, ...].detach().numpy(), 2)
            # mov_image_ref = mov_image_ref/np.max(mov_image_ref)
            for x in range(-6, 7):
                for y in range(-6, 7):
                    mov_image = np.roll(mov_image_ref, x, axis=0)
                    mov_image = np.roll(mov_image, y, axis=1)
                    mse[x+5, y+5] = np.mean((ref_image[10:-10, 10:-10] -
                                             mov_image[10:-10, 10:-10])**2)
            min_pos = np.where(mse == np.min(mse))
            shift = ((min_pos[1][0]-5)/2, (min_pos[0][0]-5)/2)
            self.notify(f'shift {d} = ({shift[0]}, {shift[1]})')
            image_out[d, ...] = self._translate_detector(image[d, ...],
                                                         (shift[0] / 2, shift[1] / 2))
            self.progress(100)
        return image_out

    @staticmethod
    def _translate_detector(img, translate):
        """Translate one detector

        :param img: Tensor for one single detector
        :param translate: (x, y, z) or (x, y) translation vector
        :return: the co-registered tensor
        """
        img_bc = img.view((1, 1, img.shape[0], img.shape[1]))
        img_out = transforms.functional.affine(img_bc, angle=0, translate=translate, scale=1,
                        shear=[0, 0],
                        interpolation=transforms.functional.InterpolationMode.BILINEAR)
        return img_out.view(img.shape)


metadata = {
    'name': 'SRegisterMSE',
    'label': 'Translation MSE',
    'class': SRegisterMSE,
    'parameters': {
    }
}
