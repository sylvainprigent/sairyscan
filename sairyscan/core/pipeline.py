import torch
from .io import SAiryscanReader


class SAiryscanPipeline:
    """Run a reconstruction pipeline in a single image

    Parameters
    ----------
    reconstruction: SAiryscanRecFilter
        Reconstruction filter
    registration: SAiryscanRegFilter
        Registration filter
    enhancing: SAiryscanEnhanceFilter

    """
    def __init__(self, reconstruction, registration=None, enhancing=None):
        self.reconstruction = reconstruction
        self.registration = registration
        self.enhancing = enhancing

    def __call__(self, image):
        """Run the pipeline

        Parameters
        ----------
        image: ndarray
            Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images

        Returns
        -------
        ndarray: the reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D

        """
        if self.registration:
            reg_image = self.registration(image)
            if self.reconstruction.__code__.co_argcount == 2:
                rec_image = self.reconstruction(image, reg_image)
            else:
                rec_image = self.reconstruction(reg_image)
        else:
            rec_image = self.reconstruction(image)

        if self.enhancing:
            return self.enhancing(rec_image)
        return rec_image


class SAiryscanLoop:
    """Apply a SAiryscan reconstruction method on all the frames and channels

    """
    def __init__(self, filename, to_file=False, destination_filename=''):
        self.filename = filename
        self.to_file = to_file
        self.destination_filename = destination_filename
        self.reader = SAiryscanReader(filename)

    def __call__(self, method):
        """Apply the reconstruction loop

        Parameters
        ----------
        method: SAiryscanFilter or SAiryscanPipeline
            instance of airyscan reconstruction filter

        Returns
        -------
        Tensor if the input to_file is false

        """
        if self.to_file:
            return self.reconstruct_to_file(method)
        else:
            return self.reconstruct_to_tensor(method)

    def reconstruct_to_tensor(self, method):
        if self.reader.depth() > 1:
            out_data = torch.zeros((self.reader.frames(), int(self.reader.channels()/2),
                                    self.reader.depth(),
                                    self.reader.height(), self.reader.width()))
        else:
            out_data = torch.zeros((self.reader.frames(), int(self.reader.channels()/2),
                                    self.reader.height(), self.reader.width()))
        for frame_idx in range(self.reader.frames()):
            for channel_idx in range(0, self.reader.channels(), 2):
                raw_data = self.reader.data(frame_idx, channel_idx)
                out_data[frame_idx, channel_idx, ...] = method(raw_data)
        return out_data.squeeze()

    def reconstruct_to_file(self, method):
        # if frame=channels=1 save to file
        # elif frame=1 and channels>1 save one file per channel filename_c1.tif, filename_c2.tif
        # else save to multiple files:
        # - filename=directory
        # - one main file per channel: filename_c1.txt and sub files filename_c1_txx.tif
        raise NotImplementedError('SAiryscanLoop: reconstruct_to_file() not yet implemented')
