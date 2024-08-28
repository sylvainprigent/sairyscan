"""This module implements the Spitfire workflow for denoising or deconvolution"""
import numpy as np

from scipy.signal import convolve2d
import torch
import torch.nn.functional as F

from .interface import SAiryscanReconstruction


def hv_loss(img: torch.Tensor, weighting: float) -> torch.Tensor:
    """Sparse Hessian regularization term

    :param img: Tensor of shape BCYX containing the estimated image,
    :param weighting: Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse,
    :return: The regularization value
    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    hv = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                    (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1])).sum()
    return hv / (a * b * h * w)


def hessian(img: torch.Tensor) -> torch.Tensor:
    """Apply Hessian filter on a tensor

    :param img: Input image tensor,
    :return: Tensor of filtered image
    """
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    return F.pad(dxx2 + dyy2 + 2 * dxy2, (1, 1, 1, 1), "constant", 0)


def dataterm_flow(blurry_image: torch.Tensor,
                  deblurred_image: torch.Tensor,
                  detector_weights: torch.Tensor
                  ) -> torch.Tensor:
    """Denoising L2 data-term

    Compute the L2 error between the original image and the reconstructed image flow

    :param blurry_image: Tensor of shape BCHYX containing the original blurry image
    :param deblurred_image: Tensor of shape BCYX containing the estimated deblurred image
    :param detector_weights: Weight applied to each detector
    """
    mse = torch.nn.MSELoss()
    mse_ = 0
    for i in range(blurry_image.shape[2]):
        mse_ += mse(blurry_image[:, :, i, :, :] -
                    detector_weights[i]*hessian(blurry_image[:, :, i, :, :]), deblurred_image)
    return mse_


def estimate_noise(data: torch.Tensor) -> float:
    """ Estimate the RMS noise of an image

    from http://stackoverflow.com/questions/2440504/
                noise-estimation-noise-measurement-in-image

    Reference: J. Immerkaer, “Fast Noise Variance Estimation”,
    Computer Vision and Image Understanding,
    Vol. 64, No. 2, pp. 300-302, Sep. 1996 [PDF]

    :param data: Image to estimate noise
    :return: The estimated variance of the noise
    """
    data = data.detach().numpy()[0, 0, ...]
    print('data.shape=', data.shape)
    height, width = data.shape
    data = np.nan_to_num(data)
    mat = [[1, -2, 1],
           [-2, 4, -2],
           [1, -2, 1]]

    sigma = np.sum(np.sum(np.abs(convolve2d(data, mat))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (width - 2) * (height - 2))

    return sigma*sigma


def calculate_weights(image: torch.Tensor) -> torch.Tensor:
    """Calculate the hessian weights of each detector

    :param image: Detectors raw images
    :return: vector containing one weight per detector
    """
    weights = torch.ones((32,))
    for d in range(32):
        weights[d] = estimate_noise(image[:, :, d, :, :])
    return weights


class SSpitfireFlow(SAiryscanReconstruction):
    """Workflow for spitfire denoising

    :param weight: Spitfire weight in ]0, 1[,
    :param reg: Spitfire regularization in ]0, 1[
    """
    def __init__(self, weight: float = 0.6, reg: float = 0.5):
        super().__init__()
        self.num_args = 1
        self.weight = weight
        self.reg = reg
        self.niter_ = 0
        self.max_iter_ = 2000
        self.gradient_step_ = 0.01
        self.loss_ = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Reconstruct the opticalflow denoised ISM with spitfire regularisation

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images
        :return: The reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D
        """
        if image.ndim == 3:
            return self.run_2d(image)
        if image.ndim == 4:
            raise NotImplementedError('Spitfire 3D deconvolution is not yet implemented')
        raise NotImplementedError('Spitfire reconstruction: image dimension not supported')

    def run_2d(self, image: torch.Tensor) -> torch.Tensor:
        """Implements the 2D case

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images
        :return: The reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D
        """
        image = image.view(1, 1, image.shape[0], image.shape[1], image.shape[2])
        deconv_image = image[:, :, 0, :, :].detach().clone()
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0
        detector_weights_ = calculate_weights(image)
        loss = None
        for _ in range(self.max_iter_):
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_flow(image, deconv_image, detector_weights_) + \
                (1-self.reg) * hv_loss(deconv_image, self.weight)
            print('iter:', self.niter_, ' loss:', loss)
            if abs(loss - previous_loss) < 1e-7:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss
        return self._crop(deconv_image.view(image.shape[3], image.shape[4]))


metadata = {
    'name': 'SSpitfireFlow',
    'label': 'Spitfire Flow',
    'class': SSpitfireFlow,
    'parameters': {
        'weight': {
            'type': float,
            'label': 'Weight',
            'help': 'Model weight between hessian and sparsity. Value is in  ]0, 1[',
            'default': 0.6,
            'range': (0, 1)
        },
        'reg': {
            'type': float,
            'label': 'Regularization',
            'help': 'Regularization weight. Value is in [0, 1]',
            'default': 0.5,
            'range': (0, 1)
        }
    }
}
