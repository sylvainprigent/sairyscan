import torch
import torchvision.transforms as transforms


class RegisterPosition:
    """Register the detectors stack by translating each image to the detector position is array

    Parameters
    ----------
    weight: int
        Weight on the detector position distance

    """
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, image):
        image_out = torch.zeros(image.shape, dtype=torch.float32)
        d = self.weight
        image_out[0, ...] = image[0, ...]
        image_out[1, ...] = self._translate_detector(image[1, ...], (d, 0.5 * d))
        image_out[2, ...] = self._translate_detector(image[2, ...], (d, -0.5 * d))
        image_out[3, ...] = self._translate_detector(image[3, ...], (0, -d))
        image_out[4, ...] = self._translate_detector(image[4, ...], (-d, -0.5 * d))
        image_out[5, ...] = self._translate_detector(image[5, ...], (-d, 0.5 * d))
        image_out[6, ...] = self._translate_detector(image[6, ...], (0, d))
        image_out[7, ...] = self._translate_detector(image[7, ...], (d, 1.5 * d))
        image_out[8, ...] = self._translate_detector(image[8, ...], (2 * d, d))
        image_out[9, ...] = self._translate_detector(image[9, ...], (2 * d, 0))
        image_out[10, ...] = self._translate_detector(image[10, ...], (2 * d, -d))
        image_out[11, ...] = self._translate_detector(image[11, ...], (d, -1.5 * d))
        image_out[12, ...] = self._translate_detector(image[12, ...], (0, -2 * d))
        image_out[13, ...] = self._translate_detector(image[13, ...], (-d, -1.5 * d))
        image_out[14, ...] = self._translate_detector(image[14, ...], (-2 * d, -d))
        image_out[15, ...] = self._translate_detector(image[15, ...], (-2 * d, 0))
        image_out[16, ...] = self._translate_detector(image[16, ...], (-2 * d, d))
        image_out[17, ...] = self._translate_detector(image[17, ...], (-d, 1.5 * d))
        image_out[18, ...] = self._translate_detector(image[18, ...], (0, 2 * d))
        image_out[19, ...] = self._translate_detector(image[19, ...], (d, 2.5 * d))
        image_out[20, ...] = self._translate_detector(image[20, ...], (2 * d, 2 * d))
        image_out[21, ...] = self._translate_detector(image[21, ...], (3 * d, 0.5 * d))
        image_out[22, ...] = self._translate_detector(image[22, ...], (3 * d, -0.5 * d))
        image_out[23, ...] = self._translate_detector(image[23, ...], (2 * d, -2 * d))
        image_out[24, ...] = self._translate_detector(image[24, ...], (d, -2.5 * d))
        image_out[25, ...] = self._translate_detector(image[25, ...], (-d, -2.5 * d))
        image_out[26, ...] = self._translate_detector(image[26, ...], (-2 * d, -2 * d))
        image_out[27, ...] = self._translate_detector(image[27, ...], (-3 * d, -0.5 * d))
        image_out[28, ...] = self._translate_detector(image[28, ...], (-3 * d, 0.5 * d))
        image_out[29, ...] = self._translate_detector(image[29, ...], (-2 * d, 2 * d))
        image_out[30, ...] = self._translate_detector(image[30, ...], (-1 * d, 2.5 * d))
        image_out[31, ...] = self._translate_detector(image[31, ...], (0, 3 * d))
        return image_out

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
