"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from os import rename
from os.path import join

from sponge_bob_magic.data_loader.archives import extract, rm_if_exists
from sponge_bob_magic.data_loader.loaders import download_dataset, download_url

logging.getLogger().setLevel(logging.INFO)

def download_movielens(path: str = ".", dataset: str = "ml-latest-small"):
    """
    Скачать датасет с https://grouplens.org/datasets/movielens/
    Доступные варианты: ml-20m, ml-latest-small, ml-latest и другие, смотри на
    сайте.

    :param path: куда положить
    :param dataset: версия мувиленса
    :return: None
    """

    logging.info("Downloading %s from grouplens...", dataset)
    archive = dataset + ".zip"
    path = join(path, archive)
    url = f"http://files.grouplens.org/datasets/movielens/{archive}"
    download_dataset(url, path)


def download_rekko(path: str = "."):
    """
    Скачать датасет с rekko chalenge
    https://boosters.pro/championship/rekko_challenge/data
    175MB

    :param path: куда положить
    :return:  None
    """
    logging.info("Downloading rekko challenge dataset...")
    archive = "rekko_challenge_rekko_challenge_2019.zip"
    url = f"https://boosters.pro/api/ch/files/pub/{archive}"
    path = join(path, "rekko.zip")
    download_dataset(url, path)


def download_netflix(path: str = "."):
    """
    Cкачать датасет Netflix Prize
    https://www.kaggle.com/netflix-inc/netflix-prize-data
    :param path: куда положить
    :return: None
    """
    logging.info("Downloading Netflix Prize dataset...")
    url = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"
    download_dataset(url, join(path, "netflix.tar.gz"))
    rename(join(path, "download"), join(path, "netflix"))
    archive = join(path, "netflix", "training_set.tar")
    extract(archive)
    rm_if_exists(archive)


def download_msd(path: str = "."):
    """
    Скачать Million Song Dataset (Echo Nest Taste Profile Subset)
    http://millionsongdataset.com/
    Данная функция скачивает тройки для обучения,
    набор данных для теста с MSD Challenge Kaggle (http://millionsongdataset.com/challenge/)
    и список песен, которые неправильно матчатся с аудио-данными
    http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/
    :param path: куда положить
    :return: None
    """
    logging.info("Getting Million Song Dataset...")
    logging.info("Downloading Echo Nest Taste Subprofile train data...")
    url = "http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip"
    download_dataset(url, join(path, "train.zip"))
    msd_folder = join(path, "msd")
    rename(join(path, "train"), msd_folder)

    logging.info("Downloading evaluation data for MSD Challenge...")
    url = "http://millionsongdataset.com/sites/default/files/challenge/EvalDataYear1MSDWebsite.zip"
    download_dataset(url, join(msd_folder, "eval.zip"))
    rename(join(msd_folder, "EvalDataYear1MSDWebsite"), join(msd_folder, "evaluation"))

    logging.info("Downloading list of matching errors...")
    url = "http://millionsongdataset.com/sites/default/files/tasteprofile/sid_mismatches.txt"
    download_url(url, join(msd_folder, "sid_mismatches.txt"))


