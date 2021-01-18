from typing import Optional

from pyspark.sql import DataFrame


# pylint: disable=too-few-public-methods
class SberDataset:
    """ Парсер набора данных из Сбера """

    def __init__(self, folder: str):
        """

        >>> data = SberDataset("")
        >>> data.log
        Traceback (most recent call last):
           ...
        ValueError: в этом наборе данных нет лога
        >>> data.user_features
        Traceback (most recent call last):
           ...
        ValueError: в этом наборе данных нет свойств пользователей
        >>> data.item_features
        Traceback (most recent call last):
           ...
        ValueError: в этом наборе данных нет свойств объектов

        :param folder: полный путь до папки с данными
        """
        self.folder = folder
        self._log: Optional[DataFrame] = None
        self._user_features: Optional[DataFrame] = None
        self._item_features: Optional[DataFrame] = None

    @property
    def log(self) -> DataFrame:
        """ лога действий пользователей """
        raise ValueError("в этом наборе данных нет лога")

    @property
    def user_features(self) -> DataFrame:
        """ свойства пользователей """
        raise ValueError("в этом наборе данных нет свойств пользователей")

    @property
    def item_features(self) -> DataFrame:
        """ свойства объектов """
        raise ValueError("в этом наборе данных нет свойств объектов")
