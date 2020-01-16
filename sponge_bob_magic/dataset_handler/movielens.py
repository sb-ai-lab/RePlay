import os
from os.path import join

from sponge_bob_magic.data_loader.datasets import download_movielens
from sponge_bob_magic.dataset_handler import DATA_FOLDER


class MovieLens():
    """
    Враппер для мувиленса, обеспечивает загрузку и парсинг данных.
    Доступны следующие размеры датасета:
    | Версия | Размер | Оценок | Пользователей | Фильмов | Тэгов |
    | :----: | :----: | :----: | :-----------: | :-----: | :---: |
    |  25m   | 250MB  |  25m   |     162k      |   62k   |  1m   |
    |  20m   | 190MB  |  20m   |     138k      |   27k   | 456k  |
    |  10m   |  63MB  |  10m   |      72k      |   10k   | 100k  |
    |   1m   |  6MB   |   1m   |      6k       |   4k    |   -   |
    |  100k  |  5MB   |  100k  |      1k       |  1.7k   |   -   |

    Подробнее: https://grouplens.org/datasets/movielens/
    """
    def __init__(self, version):
        """
        :param version: Конкретный вариант датасета
        """
        options = {'100k', '1m', '10m', '20m', '25m', 'small'}
        if version not in options:
            raise ValueError(f'{version} is not supported. Available options: {options}')

        if version == 'small':
            dataset = "ml-latest-small"
        else:
            dataset = "ml-" + version

        folder = join(DATA_FOLDER, dataset)
        if not os.path.exists(folder):
            download_movielens(DATA_FOLDER, dataset)
