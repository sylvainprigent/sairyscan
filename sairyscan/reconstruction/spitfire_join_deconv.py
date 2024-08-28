"""This module implements the Spitfire join deconvolution reconstruction method"""
import torch
from .interface import SAiryscanReconstruction
from ._detectors_weights import SAiryscanWeights


def hv_loss(img: torch.Tensor, weighting: float) -> torch.Tensor:
    """Sparse Hessian regularization term (or loss)

    :param img: Tensor of shape BCYX containing the estimated image
    :param weighting: Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse
    :return: The loss value
    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    hv = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                    (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1])).sum()
    return hv / (a * b * h * w)


def dataterm_deconv(blurry_image: torch.Tensor,
                    deblurred_image: torch.Tensor,
                    psf: torch.Tensor,
                    detector_weights: torch.Tensor
                    ) -> torch.Tensor:
    """Deconvolution L2 data-term

    Compute the L2 error between the original image and the convoluted reconstructed image

    :param blurry_image: Tensor of shape BCHYX containing the original blurry image
    :param deblurred_image: Tensor of shape BCYX containing the estimated deblurred image
    :param psf:  containing the point spread function
    :param detector_weights: Weight applied to each detector
    """
    a, b, h, w = deblurred_image.size()
    conv_op = torch.nn.Conv2d(1, 1, kernel_size=psf.shape[2],
                              stride=1,
                              padding=int((psf.shape[2] - 1) / 2),
                              bias=False)
    with torch.no_grad():
        conv_op.weight = torch.nn.Parameter(psf)
    # mse = torch.nn.MSELoss()
    mse_ = 0
    conv_img = conv_op(deblurred_image)
    for i in range(blurry_image.shape[2]):
        mse_ += detector_weights[i]*torch.sum(torch.square(blurry_image[:, :, i, :, :] - conv_img))
    return mse_ / (a * b * h * w)


class SpitfireJoinDeconv(SAiryscanReconstruction):
    """Gray scaled image deconvolution with the Spitfire algorithm

    :param psf:  Point spread function
    :param weight: Model weight between hessian and sparsity. Value is in  ]0, 1[
    :param reg: Regularization weight. Value is in [0, 1]
    :param detector_weights: Method to compute the detectors weights
    """
    def __init__(self,
                 psf: torch.Tensor,
                 weight: float = 0.6,
                 reg: float = 0.5,
                 detector_weights: str = 'mean'):
        super().__init__()
        self.num_args = 1
        self.psf = psf
        self.weight = weight
        self.reg = reg
        self.detector_weights = detector_weights
        self.niter_ = 0
        self.max_iter_ = 2000
        self.gradient_step_ = 0.01
        self.loss_ = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Reconstruct the spitfire join deconvolution

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images,
        :return: The reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D
        """
        if image.ndim == 3:
            return self.run_2d(image)
        if image.ndim == 4:
            raise NotImplementedError('Spitfire 3D deconvolution is not yet implemented')
        raise NotImplementedError('Spitfire reconstruction: image dimension not supported')

    def run_2d(self, image) -> torch.Tensor:
        """Do the 2D reconstruction

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images,
        :return: The reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D
        """
        self.progress(0)

        # detectors weights
        detector_weights_filter = SAiryscanWeights(self.detector_weights)
        detector_weights_ = detector_weights_filter()

        # normalise image
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        # deconv ref image
        deconv_image = torch.zeros(image.shape[1], image.shape[2])
        for d in range(32):
            deconv_image = deconv_image + detector_weights_[d]*image[d, ...]
        # ism_image = torch.mean(image, axis=0)
        deconv_image = deconv_image.view(1, 1, deconv_image.shape[0], deconv_image.shape[1])
        deconv_image.requires_grad = True

        image = image.view(1, 1, image.shape[0], image.shape[1], image.shape[2])
        self.psf = self.psf.view(1, 1, self.psf.shape[0], self.psf.shape[1])

        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0

        loss = None
        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_deconv(image, deconv_image, self.psf, detector_weights_) + \
                (1-self.reg) * hv_loss(deconv_image, self.weight)
            print('iter:', self.niter_, ' loss:', loss.item())
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
        self.progress(100)
        return self._crop(deconv_image.view(image.shape[3], image.shape[4]))


metadata = {
    'name': 'SpitfireJoinDeconv',
    'label': 'Spitfire Join Deconv',
    'class': SpitfireJoinDeconv,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'detector_weights': {
            'type': 'select',
            'label': 'Detectors weights',
            'help': 'Model defining weight for each detectors',
            'values': ['mean', 'ring', 'ring_inv', 'd2c', 'id2c', 'exp_d2c', 'exp_d2c_inv'],
            'default': 'mean'
        },
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
            'default': 0.995,
            'range': (0, 1)
        }
    }
}
