from .interface import SairyscanRegistration
from .position import SRegisterPosition
from .fourier_phase import SRegisterFourierPhase
from .mse_registration import SRegisterMSE

from .position import metadata as position_metadata
from .mse_registration import metadata as mse_registration_metadata


metadata = [position_metadata, mse_registration_metadata]

__all__ = ['SairyscanRegistration', 'SRegisterPosition', 'SRegisterFourierPhase', 'SRegisterMSE', 'metadata']
