"""Module to define the processing class interface using the observer pattern"""
from abc import ABC, abstractmethod
import torch


class SObserver(ABC):
    """Interface of observer to notify progress

    An observer must implement the progress and message

    """
    def __init__(self):
        pass

    @abstractmethod
    def notify(self, message: str):
        """Notify a progress message

        :param message: Progress message
        """
        raise NotImplementedError('SObserver is abstract')

    @abstractmethod
    def progress(self, value: int):
        """Notify progress value

        :param value: Progress value in [0, 100]
        """
        raise NotImplementedError('SObserver is abstract')


class SObservable:
    """Interface for any data processing class

    The observable class can notify the observers for progress
    """
    def __init__(self):
        self._observers = []

    def add_observer(self, observer: SObserver):
        """Add an observer

        :param observer: Observer instance

        """
        self._observers.append(observer)

    def notify(self, message: str):
        """Notify progress to observers

        :param message: Progress message
        """
        for obs in self._observers:
            obs.notify(message)

    def progress(self, value: int):
        """Notify progress to observers

        :param value: Progress value in [0, 100]
        """
        for obs in self._observers:
            obs.progress(value)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Run the data processing

        :param image: Image to process,
        :return: The processed image
        """


class SObserverConsole(SObserver):
    """Implementation of an observer that print messages and progress to console"""
    def notify(self, message):
        print(message)

    def progress(self, value):
        print('progress:', value, '%')
