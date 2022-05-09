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
        if image.ndims == 2:
            fft_source = torch.fft.fft2(image)
            psf_roll = torch.roll(self.psf, int(-self.psf.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-self.psf.shape[1] / 2), dims=1)
            fft_psf = torch.fft.fft2(psf_roll)

            fft_psf_mirror = torch.fft.fft2(torch.flip(self.psf, int(-self.psf.shape[0] / 2), dims=0))

            fft_out = fft_source
            out_image = image.copy()
            for i in range(self.niter):
                fft_tmp = fft_out*fft_psf
                tmp = torch.fft.ifft2(fft_tmp)
                tmp = image/tmp
                fft_tmp = torch.fft.fft2(tmp)
                fft_tmp = fft_tmp * fft_psf_mirror
                tmp = torch.fft.ifft2(fft_tmp)
                out_image = out_image * tmp
            return out_image

        elif image.ndims == 3:
            raise NotImplementedError('Richardson Lucy ED deconvolution is not yet implemented')
