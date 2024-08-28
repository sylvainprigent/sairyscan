"""This module implements a pipeline maker tool for SAiryscan"""
import torch
from sairyscan.core.io import SAiryscanReader
from sairyscan.core import SObservable
from sairyscan.registration import SAiryscanRegistration
from sairyscan.reconstruction import SAiryscanReconstruction
from sairyscan.enhancing import SAiryscanEnhancing


class SAiryscanPipeline(SObservable):
    """Build and run a reconstruction pipeline in a single image

    :param reconstruction: instance of the reconstruction filter
    :param registration: instance of the registration filter
    :param enhancing: instance of the enhancing filter
    """
    def __init__(self,
                 reconstruction: SAiryscanReconstruction,
                 registration: SAiryscanRegistration = None,
                 enhancing: SAiryscanEnhancing = None):
        super().__init__()
        self.reconstruction = reconstruction
        self.registration = registration
        self.enhancing = enhancing

    def _transmit_observers(self, child: SObservable):
        """Transmit observer to the observable

        :param child: Object that will receive the observers
        """
        for observer in self._observers:
            child.add_observer(observer)

    def __call__(self, image: torch.Tensor):
        """Run the pipeline

        :param image: Raw airyscan image. [H, Z, Y, X] for 3D image, [H, Y, X] for 2D images,
        :return: The reconstructed image. [Z, Y, X] for 3D, [Y, X] for 2D
        """
        self._transmit_observers(self.reconstruction)
        if self.registration:
            self._transmit_observers(self.registration)
            reg_image = self.registration(image)
            if self.reconstruction.num_args == 2:
                rec_image = self.reconstruction(image, reg_image)
            else:
                rec_image = self.reconstruction(reg_image)
        else:
            rec_image = self.reconstruction(image)

        if self.enhancing:
            self._transmit_observers(self.enhancing)
            return self.enhancing(rec_image)
        return rec_image


class SAiryscanLoop(SObservable):
    """Apply a SAiryscan reconstruction method on all the frames and channels

    :param filename: Path to the file containing the data,
    :param to_file: True to dave the tensor to file. Otherwise, the call method return a tensor
    :param destination_filename: Path to the file where the result is saved
    """
    def __init__(self, filename: str, to_file=False, destination_filename=''):
        super().__init__()
        self.filename = filename
        self.to_file = to_file
        self.destination_filename = destination_filename
        self.reader = SAiryscanReader(filename)

    def __call__(self, method: SObservable) -> torch.Tensor | None:
        """Apply the reconstruction loop

        :param method: instance of airyscan filter or pipeline
        :return: a torch.Tensor if the input to_file is false
        """
        if self.to_file:
            return self.reconstruct_to_file(method)
        return self.reconstruct_to_tensor(method)

    def reconstruct_to_tensor(self, method: SObservable) -> torch.Tensor:
        """Implement the reconstruction to a tensor

        :param method: instance of airyscan filter or pipeline
        :return: a torch.Tensor with the processed image
        """
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

    def reconstruct_to_file(self, method: SObservable):
        """Reconstruct an image directly into a file (for large memory data)

        TODO: Implement this method

        :param method: instance of airyscan filter or pipeline
        """
        raise NotImplementedError('SAiryscanLoop: reconstruct_to_file() not yet implemented')
