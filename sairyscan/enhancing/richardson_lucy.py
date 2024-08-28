"""This module implements the Richardson-Lucy deconvolution algorithm"""
import torch
from .interface import SAiryscanEnhancing


class SAiryscanRichardsonLucy(SAiryscanEnhancing):
    """Apply a Richardson-Lucy deconvolution

    :param psf: Point spread function
    :param niter: Number of iterations
    """
    def __init__(self, psf: torch.Tensor, niter: int = 30):
        super().__init__()
        self.psf = psf
        self.niter = niter

    @staticmethod
    def _resize_psf(psf: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Pad a PSF image to match a given shape

        :param psf: The Point Spread Function to resize,
        :param width: Targeted image width,
        :param height: Targeted image height,
        :return: The resized PSF image
        """
        kernel = torch.zeros((width, height))
        x_start = int(width / 2 - psf.shape[0] / 2) + 1
        y_start = int(height / 2 - psf.shape[1] / 2) + 1
        kernel[x_start:x_start+psf.shape[0], y_start:y_start+psf.shape[1]] = psf
        return kernel

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the deconvolution

        :param image: Image to deblur [Y, X] or [Z, Y, X]
        :return: The deblurred image
        """
        if image.ndim == 2:
            padding = 13
            psf = self._resize_psf(self.psf, image.shape[0]+2*padding, image.shape[1]+2*padding)
            psf_roll = torch.roll(psf, [int(-psf.shape[0]/2),
                                        int(-psf.shape[1]/2)], dims=(0, 1))
            fft_psf = torch.fft.fft2(psf_roll)
            fft_psf_mirror = torch.fft.fft2(torch.flip(psf_roll, dims=[0, 1]))

            pad_fn = torch.nn.ReflectionPad2d(padding)
            image_pad = pad_fn(image.detach().clone().view(1, 1,
                            image.shape[0], image.shape[1])).view((image.shape[0]+2*padding,
                                                                   image.shape[1]+2*padding))
            out_image = pad_fn(image.detach().clone().view(1, 1,
                            image.shape[0], image.shape[1])).view((image.shape[0]+2*padding,
                                                                   image.shape[1]+2*padding))
            for _ in range(self.niter):
                fft_out = torch.fft.fft2(out_image)
                fft_tmp = fft_out*fft_psf
                tmp = torch.real(torch.fft.ifft2(fft_tmp))
                tmp = image_pad/tmp
                fft_tmp = torch.fft.fft2(tmp)
                fft_tmp = fft_tmp * fft_psf_mirror
                tmp = torch.real(torch.fft.ifft2(fft_tmp))
                out_image = out_image * tmp
            return out_image[padding:-padding, padding:-padding]

        raise NotImplementedError('Richardson Lucy 3D deconvolution is not yet implemented')


metadata = {
    'name': 'SAiryscanRichardsonLucy',
    'label': 'Richardson-Lucy',
    'class': SAiryscanRichardsonLucy,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'niter': {
            'type': int,
            'label': 'niter',
            'help': 'Number of iterations',
            'default': 30,
            'range': (0, 999999)
        }
    }
}
