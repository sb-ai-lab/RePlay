import os
from os.path import join
import pandas as pd
from pandas import DataFrame
from typing import Tuple

from sponge_bob_magic.datasets.data_loader import download_movielens
from sponge_bob_magic.datasets.generic_dataset import Dataset


class MovieLens(Dataset):
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

    Прочитанные объекты доступны как атрибуты класса:
    - ratings -- лог оценок фильмов пользователями
    - items -- фичи фильмов
    - users -- фичи пользователей
    - tags -- лог тегов, которые пользователи ставили фильмам
    - links -- связка id датасета с imdb и tmdb
    - genome_tags -- список тегов из tag genome dataset
    - genome_scores -- теги с важностью

    Только некоторые атрибуты доступны для каждой версии датасета.
    Это определяется файлами в датасете,
    например, начиная с 10m отсутствуют фичи пользователей,
    а начиная с 20m можно дополнительно прочитать tag genome dataset.

    Посмотреть доступные данные можно с помощью метода `info()`.

    Пример загрузки:
    >>> from sponge_bob_magic.datasets import MovieLens
    >>> ml = MovieLens("10m")
    >>> ml.info()
    ratings
       user_id  item_id  relevance  timestamp
    0        1      122        5.0  838985046
    1        1      185        5.0  838983525
    2        1      231        5.0  838983392
    items
       item_id                    title  \
    0        1         Toy Story (1995)
    1        2           Jumanji (1995)
    2        3  Grumpier Old Men (1995)
                                            genres
    0  Adventure|Animation|Children|Comedy|Fantasy
    1                   Adventure|Children|Fantasy
    2                               Comedy|Romance
    tags
       user_id  item_id         tag   timestamp
    0       15     4973  excellent!  1215184630
    1       20     1747    politics  1188263867
    2       20     1747      satire  1188263867

    Подробнее: https://grouplens.org/datasets/movielens/
    """
    def __init__(self, version: str = "small", read_genome: bool = False, path: str = None):
        """
        :param version: Конкретный вариант датасета
        :param read_genome: Читать ли данные genome tag dataset (если включены в датасет),
            по умолчанию не читаются для экономии памяти.
        :param path: где искать и куда класть датасет.
        """
        super().__init__(path)
        options = {"100k", "1m", "10m", "20m", "25m", "small", "latest"}
        if version not in options:
            raise ValueError(f"{version} is not supported. Available options: {options}")

        if version == "small":
            dataset = "ml-latest-small"
        else:
            dataset = "ml-" + version

        folder = join(self.data_folder, dataset)
        if not os.path.exists(folder):
            download_movielens(self.data_folder, dataset)

        if version == "100k":
            self.ratings, self.users, self.items = self._read_100k(folder)
        elif version == "1m":
            self.ratings, self.users, self.items = self._read_1m(folder)
        elif version == "10m":
            self.ratings, self.items, self.tags = self._read_10m(folder)
        else:
            self.ratings, self.items, self.tags, self.links = self._read_modern(folder)
            if read_genome:
                self.genome_tags, self.genome_scores = self._read_genome(folder)

    @staticmethod
    def _read_modern(folder: str) -> Tuple[DataFrame]:
        ratings = pd.read_csv(join(folder, "ratings.csv"), header=0,
                              names=["user_id", "item_id", "relevance", "timestamp"])
        items = pd.read_csv(join(folder, "movies.csv"), header=0,
                            names=["item_id", "title", "genres"])
        tags = pd.read_csv(join(folder, "tags.csv"), header=0,
                           names=["user_id", "item_id", "tag", "timestamp"])
        links = pd.read_csv(join(folder, "links.csv"), header=0,
                            names=["item_id", "imdb_id", "tmdb_id"])
        return ratings, items, tags, links

    @staticmethod
    def _read_genome(folder: str) -> Tuple[DataFrame]:
        genome_tags = pd.read_csv(join(folder, "genome-tags.csv"), header=0,
                                  names=["tag_id", "tag"])
        genome_scores = pd.read_csv(join(folder, "genome-scores.csv"), header=0,
                                    names=["movie_id", "tag_id", "relevance"])
        return genome_tags, genome_scores

    @staticmethod
    def _read_10m(folder: str) -> Tuple[DataFrame]:
        ratings = pd.read_csv(join(folder, "ratings.dat"), sep="\t",
                              names=["user_id", "item_id", "relevance", "timestamp"])
        items = pd.read_csv(join(folder, "movies.dat"), sep="\t",
                            names=["item_id", "title", "genres"])
        tags = pd.read_csv(join(folder, "tags.dat"), sep="\t",
                           names=["user_id", "item_id", "tag", "timestamp"])
        return ratings, items, tags

    @staticmethod
    def _read_1m(folder: str) -> Tuple[DataFrame]:
        ratings = pd.read_csv(join(folder, "ratings.dat"), sep="\t",
                              names=["user_id", "item_id", "relevance", "timestamp"])
        users = pd.read_csv(join(folder, "users.dat"), sep="\t",
                            names=["user_id", "gender", "age", "occupation", "zip_code"])
        items = pd.read_csv(join(folder, "movies.dat"), sep="\t",
                            names=["item_id", "title", "genres"])
        return ratings, users, items

    @staticmethod
    def _read_100k(folder: str) -> Tuple[DataFrame]:
        ratings = pd.read_csv(join(folder, "u.data"), sep="\t",
                              names=["user_id", "item_id", "relevance", "timestamp"])
        users = pd.read_csv(join(folder, "u.user"), sep="|",
                            names=["user_id", "gender", "age", "occupation", "zip_code"])
        items = pd.read_csv(join(folder, "u.item"), sep="|",
                            names=["item_id", "title", "release_date", "video_release_date",
                                   "imdb_url", "unknown", "Action", "Adventure", "Animation",
                                   "Children\'s", "Comedy", "Crime", "Documentary", "Drama",
                                   "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                                   "Romance", "Sci-Fi", "Thriller", "War", "Western"],
                            encoding="ISO-8859-1",
                            parse_dates=["release_date"]).drop("video_release_date", axis=1)
        return ratings, users, items
