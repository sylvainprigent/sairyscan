"""Airyscan IO

Set of IO methods to read Airyscan raw files

Classes
-------
AiryscanReader

"""
from pathlib import Path
import numpy as np
import torch
from aicspylibczi import CziFile
from .settings import Settings


class SAiryscanReader:
    """Read a CZI Zeiss raw data

    Parameters
    ----------
    filename: str
        Path to the .czi file

    """
    def __init__(self, filename):
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

    def frames(self):
        """Get the number of frames in the CZI file"""
        return self.dimensions['T'][1]

    def channels(self):
        """Get the number of channels in the CZI file"""
        return self.dimensions['C'][1]

    def depth(self):
        """Get the number of slices in the CZI file"""
        return self.dimensions['Z'][1]

    def width(self):
        """Get the number of columns in the CZI file"""
        return self.dimensions['X'][1]

    def height(self):
        """Get the number of rows in the CZI file"""
        return self.dimensions['Y'][1]

    def to_tensor(self):
        data = np.zeros((self.frames(), ))
        for frame in self.frames():
            for channel in self.channels():
                pass

    def data(self, t=0, c=0):
        """Extract one frame from the CZI data

        Parameters
        ----------
        t: int
            Frame index
        c: int
            Channel index

        Returns
        -------
        a numpy array [H, Z, Y, X] for 3D slice, [H, Y, X] for 2D slices. Where H is the detector
        dimension

        """
        img, shp = self.czi.read_image(T=t, C=c)
        return torch.from_numpy(np.float32(np.squeeze(img))).to(Settings.instance().device)
