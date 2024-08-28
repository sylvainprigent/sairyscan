"""Registration module"""
from .interface import SAiryscanRegistration
from .position import SRegisterPosition
from .fourier_phase import SRegisterFourierPhase
from .mse_registration import SRegisterMSE

from .position import metadata as position_metadata
from .mse_registration import metadata as mse_registration_metadata


metadata = [position_metadata, mse_registration_metadata]

__all__ = [
    'SAiryscanRegistration',
    'SRegisterPosition',
    'SRegisterFourierPhase',
    'SRegisterMSE',
    'metadata'
]
