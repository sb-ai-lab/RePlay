import logging
from glob import glob
from os import mkdir, remove
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

    ! В первую загрузку чтение будет происходить долго, около 7 минут из-за парсинга дат.
    ! В последствии будет читаться быстро т.к. будет оставлен только обработанный паркет файл.

    ! Не рекомендуется читать датафрейм напрямую (df = pd.read_parquet)
    ! при использовании саентифик мод пайчарм.
    ! Пайчарм пытается прогрузить все 100млн строк, объекта,
    ! чтобы отображать информацию о датафрейме,
    ! что приводит к огромному потреблению памяти и, вероятно, зависанию.
    ! В обертке (например, этот класс) он туда не лезет,
    ! если не разворачивать атрибуты объекта вручную.
    """
    def __init__(self, path: str = None):
        super().__init__(path)
        folder = join(self.data_folder, "netflix")
        if not exists(folder):
            download_netflix(self.data_folder)
            self._save_clean(folder)
        self._read_clean(folder)

    def _read_clean(self, folder):
        path = join(folder, "clean")
        self.movies = pd.read_csv(join(path, "movies.csv"), sep="\t",
                                  names=["item_id", "year", "title"],
                                  dtype={"item_id": "uint16",
                                          "year": "float32"})
        self.test = pd.read_csv(join(path, "test.csv"),
                                names=["item_id", "user_id", "timestamp"],
                                parse_dates=["timestamp"],
                                dtype={"user_id": "category",
                                       "item_id": "category"})
        if exists(join(path, "train.parquet")):
            self.train = pd.read_parquet(join(path, 'train.parquet'))
        else:
            logging.info("One time date parsing will take ≈ 7 minutes...")
            self.train = pd.read_csv(join(path, "train.csv"),
                                     names=["item_id", "user_id", "relevance", "timestamp"],
                                     parse_dates=["timestamp"],
                                     dtype={"user_id": "category",
                                            "item_id": "category",
                                            "relevance": "uint8"})
            self.train.to_parquet(join(path, "train.parquet"))
            remove(join(path, "train.csv"))

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
