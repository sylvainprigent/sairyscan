"""This module implements image enhancing methods (denoising and deconvolution)"""
from .interface import SAiryscanEnhancing
from .gaussian import SAiryscanGaussian
from .wiener import SAiryscanWiener
from .richardson_lucy import SAiryscanRichardsonLucy
from ._psfs import PSFGaussian
from .spitfire_deconv import SpitfireDeconv

from .gaussian import metadata as gaussian_metadata
from .richardson_lucy import metadata as richardson_lucy_metadata
from .wiener import metadata as wiener_metadata
from .spitfire_deconv import metadata as spitfiredeconv_metadata
from .spitfire_denoise import metadata as spitfiredenoise_metadata


metadata = [gaussian_metadata,
            spitfiredenoise_metadata,
            richardson_lucy_metadata,
            wiener_metadata,
            spitfiredeconv_metadata
            ]


__all__ = [
    'PSFGaussian',
    'SAiryscanEnhancing',
    'SAiryscanGaussian',
    'SAiryscanWiener',
    'SAiryscanRichardsonLucy',
    'SpitfireDeconv',
    'metadata'
]
