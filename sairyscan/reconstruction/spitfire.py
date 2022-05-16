import torch
from .interface import SAiryscanReconstruction


def hv_loss(img, weighting):
    """Sparse Hessian regularization term

    Parameters
    ----------
    img: Tensor
        Tensor of shape BCYX containing the estimated image
    weighting: float
        Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse

    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    hv = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                    (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1])).sum()
    return hv / (a * b * h * w)


def dataterm_deconv(blurry_image, deblurred_image, psf, detector_weights):
    """Deconvolution L2 data-term

    Compute the L2 error between the original image and the convoluted reconstructed image

    Parameters
    ----------
    blurry_image: Tensor
        Tensor of shape BCHYX containing the original blurry image
    deblurred_image: Tensor
        Tensor of shape BCYX containing the estimated deblurred image
    psf: Tensor
        Tensor containing the point spread function
    detector_weights: Tensor
        Weight applied to each detector

    """
    conv_op = torch.nn.Conv2d(1, 1, kernel_size=psf.shape[2],
                              stride=1,
                              padding=int((psf.shape[2] - 1) / 2),
                              bias=False)
    with torch.no_grad():
        conv_op.weight = torch.nn.Parameter(psf)
    mse = torch.nn.MSELoss()
    mse_ = 0
    conv_img = conv_op(deblurred_image)
    for i in range(blurry_image.shape[2]):
        mse_ += detector_weights[i]*mse(blurry_image[:, :, i, :, :], conv_img)
    return mse_


class SpitfireReconstruction(SAiryscanReconstruction):
    """Gray scaled image deconvolution with the Spitfire algorithm

    Parameters
    ----------
    psf: Tensor
        Point spread function
    weight: float
        model weight between hessian and sparsity. Value is in  ]0, 1[
    reg: float
        Regularization weight. Value is in [0, 1]

    """

    def __init__(self, psf, weight=0.6, reg=0.5, detector_weights='mean'):
        super().__init__()
        self.psf = psf
        self.weight = weight
        self.reg = reg
        self.detector_weights = detector_weights
        self.niter_ = 0
        self.max_iter_ = 2000
        self.gradient_step_ = 0.01
        self.loss_ = None

    @staticmethod
    def weights_exp():
        tau = 3.0
        weights = torch.ones((32,))
        for i in range(1, 7):
            weights[i] = 1/tau
        for i in range(7, 19):
            weights[i] = 2 / tau
        for i in range(19, 32):
            weights[i] = 3 / tau
        return weights/torch.sum(weights)

    @staticmethod
    def weights_mean():
        weights = torch.ones((32,))
        return weights/torch.sum(weights)

    def __call__(self, image):
        """Reconstruct the spitfire join deconvolution

        Parameters
        ----------
        image: Tensor
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        Tensor: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        if image.ndim == 3:
            return self.run_2d(image)
        elif image.ndim == 4:
            raise NotImplementedError('Spitfire 3D deconvolution is not yet implemented')
        else:
            raise NotImplementedError('Spitfire reconstruction: image dimension not supported')

    def run_2d(self, image):
        image = image.view(1, 1, image.shape[0], image.shape[1], image.shape[2])
        self.psf = self.psf.view(1, 1, self.psf.shape[0], self.psf.shape[1])
        deconv_image = image[:, :, 0, :, :].detach().clone()
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0
        if self.detector_weights == 'exp':
            detector_weights_ = self.weights_exp()
        elif self.detector_weights == 'mean':
            detector_weights_ = self.weights_mean()
        for _ in range(self.max_iter_):
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_deconv(image, deconv_image, self.psf, detector_weights_) + \
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
        return deconv_image.view(image.shape[3], image.shape[4])


metadata = {
    'name': 'SpitfireReconstruction',
    'class': SpitfireReconstruction,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
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
            'default': 0.5,
            'range': (0, 1)
        }
    }
}
