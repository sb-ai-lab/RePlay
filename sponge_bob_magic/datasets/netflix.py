import logging
from glob import glob
from os import mkdir
from os.path import join, exists
import pandas as pd
from tqdm import tqdm

from sponge_bob_magic.datasets.data_loader import download_netflix
from sponge_bob_magic.datasets.generic_dataset import Dataset


class Netflix(Dataset):
    """
    Враппер для датасета с Netflix Prize, обеспечивает загрузку и парсинг данных.

    - 480,189 пользователей
    - 17,770 фильмов
    - 100m оценок по шкале 1-5

    Для тестовых данных оценки не доступны.

    Подробнее: https://www.kaggle.com/netflix-inc/netflix-prize-data
    """
    def __init__(self, path: str = None):
        super().__init__(path)
        folder = join(self.data_folder, "netflix")
        if not exists(folder):
            download_netflix(self.data_folder)
            self._save_clean(folder)
        self._read_clean(folder)

    def _read_clean(self, folder):
        logging.info("loading preprocessed")
        path = join(folder, "clean")
        self.movies = pd.read_csv(join(path, "movies.csv"), sep="\t",
                                  names=["item_id", "year", "title"],
                                  dtype={"item_id": "uint16",
                                          "year": "float32"})
        self.train = pd.read_csv(join(path, "train.csv"),
                                 names=["item_id", "user_id", "relevance", "timestamp"],
                                 parse_dates=["timestamp"],
                                 dtype={"user_id": "category",
                                        "item_id": "category",
                                        "relevance": "uint8"})
        self.test = pd.read_csv(join(path, "test.csv"),
                                names=["item_id", "user_id", "timestamp"],
                                parse_dates=["timestamp"],
                                dtype={"user_id": "category",
                                       "item_id": "category"})

    def _save_clean(self, raw):
        clean = join(raw, "clean")
        mkdir(clean)
        self._fix_movies(raw, clean)
        self._fix_train(raw, clean)
        self._fix_test(raw, clean)

    @staticmethod
    def _fix_test(raw, clean):
        dest = open(join(clean, "test.csv"), "w")
        with open(join(raw, "qualifying.txt")) as source:
            for line in source:
                if line[-2] == ":":
                    movie_id = line[:-2] + ","
                else:
                    dest.write(movie_id + line)
        dest.close()

    @staticmethod
    def _fix_train(raw, clean):
        logging.info("Parsing train files")
        folder = join(raw, "training_set")
        files = glob(join(folder, "*.txt"))
        dest = open(join(clean, "train.csv"), "w")
        for file in tqdm(files):
            with open(file) as source:
                for line in source:
                    if line[-2] == ":":
                        movie_id = line[:-2] + ","
                    else:
                        dest.write(movie_id + line)
        dest.close()

    @staticmethod
    def _fix_movies(raw, clean):
        """Разделитель запятая так же встречается в названиии фильмов, например:
        `72,1974,At Home Among Strangers, A Stranger Among His Own`
        Разделитель запятая заменяется на табуляцию, для простоты парсинга.
        """
        file = join(raw, "movie_titles.txt")
        dest = open(join(clean, "movies.csv"), "w")
        with open(file, encoding="ISO-8859-1") as f:
            for line in f.readlines():
                first = line.find(",")
                second = first + 5
                m_id = line[:first]
                year = line[first+1:second]
                title = line[second+1:]
                dest.write("\t".join([m_id, year, title]) + "\n")
        dest.close()
