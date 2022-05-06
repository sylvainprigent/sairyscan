import torch


class SettingsContainer:
    """Container for the SAiryscan library settings"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Settings:
    """Singleton to access the Settings container
        
    Raises
    ------
    Exception: if multiple instantiation of the Settings container is tried
    """
    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if Settings.__instance is not None:
            raise Exception("Settings container can be initialized only once!")
        else:
            Settings.__instance = SettingsContainer()

    @staticmethod
    def instance():
        """ Static access method to the Config. """
        if Settings.__instance is None:
            Settings.__instance = SettingsContainer()
        return Settings.__instance
