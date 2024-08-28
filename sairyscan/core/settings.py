"""This module implements the read/write of user settings"""
import torch


class SettingsContainer:
    """Container for the SAiryscan library settings"""
    def __init__(self):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> str:
        """Get the device ID

        :return: The device ID
        """
        return self.__device


class Settings:
    """Singleton to access the Settings container
        
    :raises: RuntimeError if multiple instantiation of the Settings container is tried
    """
    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if Settings.__instance is not None:
            raise RuntimeError("Settings container can be initialized only once!")
        Settings.__instance = SettingsContainer()

    @staticmethod
    def instance():
        """ Static access method to the Config. """
        if Settings.__instance is None:
            Settings.__instance = SettingsContainer()
        return Settings.__instance
