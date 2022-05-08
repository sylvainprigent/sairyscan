import torch
from .interface import SAiryscanEnhancing


class SAiryscanWiener(SAiryscanEnhancing):
    """Apply a gaussian filter

    Parameters
    ----------
    psf: Tensor
        Point spread function

    """
    def __init__(self, psf, beta=1e-5):
        super().__init__()
        self.psf = psf
        self.beta = beta

    def __call__(self, image):
        if image.ndims == 2:
            fft_source = torch.fft.fft2(image)
            psf_roll = torch.roll(self.psf, int(-self.psf.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-self.psf.shape[1] / 2), dims=1)
            fft_psf = torch.fft.fft2(psf_roll)
            return torch.real(torch.fft.ifft2(fft_source * torch.conj(fft_psf) / (
                self.beta * fft_source * torch.conj(fft_source) + fft_psf * torch.conj(fft_psf))))
        elif image.ndims == 3:
            fft_source = torch.fft.fftn(image)
            psf_roll = torch.roll(self.psf, int(-self.psf.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-self.psf.shape[1] / 2), dims=1)
            psf_roll = torch.roll(psf_roll, int(-self.psf.shape[2] / 2), dims=2)
            fft_psf = torch.fft.fftn(psf_roll)
            return torch.real(torch.fft.ifftn(fft_source * torch.conj(fft_psf) / (
                self.beta * fft_source * torch.conj(fft_source) + fft_psf * torch.conj(fft_psf))))
