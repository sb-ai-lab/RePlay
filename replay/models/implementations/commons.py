from abc import ABC, abstractmethod
from typing import Dict


class IsSavable(ABC):  # TODO: Think about remove to other module and use to PandasImpl, PolarsImpl
    """
    Common methods and attributes for saving and loading RePlay models
    """

    @property
    @abstractmethod
    def _init_args(self) -> Dict:
        """
        Dictionary of the model attributes passed during model initialization.
        Used for model saving and loading
        """

    @property
    def _dataframes(self) -> Dict:
        """
        Dictionary of the model dataframes required for inference.
        Used for model saving and loading
        """
        return {}

    @abstractmethod
    def _save_model(self, path: str) -> None:
        """
        Method for dump model attributes to disk
        """

    @abstractmethod
    def _load_model(self, path: str) -> None:
        """
        Method for loading model attributes from disk
        """
