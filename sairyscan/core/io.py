"""Airyscan IO

Set of IO methods to read Airyscan raw files
"""
from pathlib import Path
import numpy as np
import torch
from aicspylibczi import CziFile
from .settings import Settings


class SAiryscanReader:
    """Read a CZI Zeiss raw data

    :param filename: Path to the .czi file

    """
    def __init__(self, filename: str):
        super().__init__()
        if filename.endswith('.czi') or filename.endswith('.CZI'):
            self.filename = filename
        else:
            raise IOError('AiryscanReader can only read CZI files')

        pth = Path(filename)
        self.czi = CziFile(pth)
        self.dimensions = self.czi.get_dims_shape()[0]
        print('pixel type=', self.czi.pixel_type)
        print('dimensions=', self.dimensions)

    def frames(self) -> int:
        """Get the number of frames in the CZI file"""
        return self.dimensions['T'][1]

    def channels(self) -> int:
        """Get the number of channels in the CZI file"""
        return self.dimensions['C'][1]

    def depth(self) -> int:
        """Get the number of slices in the CZI file"""
        return self.dimensions['Z'][1]

    def width(self) -> int:
        """Get the number of columns in the CZI file"""
        return self.dimensions['X'][1]

    def height(self) -> int:
        """Get the number of rows in the CZI file"""
        return self.dimensions['Y'][1]

    def data(self, frame_idx: int = 0, channel_idx: int = 0):
        """Extract one frame from the CZI data

        :param frame_idx: Frame index
        :param channel_idx: Channel index
        :return: a torch array [H, Z, Y, X] for 3D slice, [H, Y, X] for 2D slices. Where H is the
                 detector dimension
        """
        img, _ = self.czi.read_image(T=frame_idx, C=channel_idx)
        return torch.from_numpy(np.float32(np.squeeze(img))).to(Settings.instance().device)
