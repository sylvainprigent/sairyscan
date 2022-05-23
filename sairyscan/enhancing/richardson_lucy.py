import torch
from .interface import SAiryscanEnhancing


class SAiryscanRichardsonLucy(SAiryscanEnhancing):
    """Apply a gaussian filter

    Parameters
    ----------
    psf: Tensor
        Point spread function

    """
    def __init__(self, psf, niter=30):
        super().__init__()
        self.psf = psf
        self.niter = niter

    def __call__(self, image):
        if image.ndim == 2:
            psf_roll = torch.roll(self.psf, [int(-self.psf.shape[0]/2),
                                             int(-self.psf.shape[1]/2)], dims=(0, 1))
            fft_psf = torch.fft.fft2(psf_roll)

            fft_psf_mirror = torch.fft.fft2(torch.flip(psf_roll, dims=[0, 1]))

            out_image = image.detach().clone()
            for i in range(self.niter):
                fft_out = torch.fft.fft2(out_image)
                fft_tmp = fft_out*fft_psf
                tmp = torch.real(torch.fft.ifft2(fft_tmp))
                tmp = image/tmp
                fft_tmp = torch.fft.fft2(tmp)
                fft_tmp = fft_tmp * fft_psf_mirror
                tmp = torch.real(torch.fft.ifft2(fft_tmp))
                out_image = out_image * tmp
            return out_image

        elif image.ndim == 3:
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
        'beta': {
            'type': int,
            'label': 'niter',
            'help': 'Number of iterations',
            'default': 30,
            'range': (0, 999999)
        }
    }
}
