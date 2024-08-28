"""This module implements the Wiener deconvolution algorithm"""
import torch
from .interface import SAiryscanEnhancing


class SAiryscanWiener(SAiryscanEnhancing):
    """Apply a gaussian filter

    :param psf: Point spread function
    :param beta: Regularization parameter

    """
    def __init__(self, psf: torch.Tensor, beta: float = 1e-5):
        super().__init__()
        self.psf = psf
        self.beta = beta

    @staticmethod
    def _resize_psf(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """Pad a PSF image to match a given shape

        :param image: Target image shape
        :param psf: The Point Spread Function to resize,
        :return: The resized PSF image
        """
        kernel = torch.zeros(image.shape)
        x_start = int(image.shape[0] / 2 - psf.shape[0] / 2) + 1
        y_start = int(image.shape[1] / 2 - psf.shape[1] / 2) + 1
        kernel[x_start:x_start+psf.shape[0], y_start:y_start+psf.shape[1]] = psf
        return kernel

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the deconvolution

        :param image: Image to deblur [Y, X] or [Z, Y, X]
        :return: THe deblurred image
        """
        if image.ndim == 2:
            padding = 13
            pad_fn = torch.nn.ReflectionPad2d(padding)
            image_pad = pad_fn(image.detach().clone().view(1, 1,
                image.shape[0], image.shape[1])).view(
                    (image.shape[0] + 2 * padding, image.shape[1] + 2 * padding))
            fft_source = torch.fft.fft2(image_pad)
            psf = self._resize_psf(image_pad, self.psf)
            psf_roll = torch.roll(psf, int(-psf.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-psf.shape[1] / 2), dims=1)
            fft_psf = torch.fft.fft2(psf_roll)
            return torch.real(torch.fft.ifft2(fft_source * torch.conj(fft_psf) / (
                 self.beta**2 + fft_psf * torch.conj(fft_psf))))[padding:-padding, padding:-padding]
        if image.ndim == 3:
            fft_source = torch.fft.fftn(image)
            psf_roll = torch.roll(self.psf, int(-self.psf.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-self.psf.shape[1] / 2), dims=1)
            psf_roll = torch.roll(psf_roll, int(-self.psf.shape[2] / 2), dims=2)
            fft_psf = torch.fft.fftn(psf_roll)
            return torch.real(torch.fft.ifftn(fft_source * torch.conj(fft_psf) / (
                self.beta * fft_source * torch.conj(fft_source) + fft_psf * torch.conj(fft_psf))))
        raise ValueError("The image shape does not match for 2D or 3D images")


metadata = {
    'name': 'SAiryscanWiener',
    'label': 'Wiener',
    'class': SAiryscanWiener,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'beta': {
            'type': float,
            'label': 'Beta',
            'help': 'Regularisation parameter',
            'default': 1e-5,
            'range': (0, 999999)
        }
    }
}
