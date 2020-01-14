"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from os import rename, mkdir
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


def download_pinterest(path: str = "."):
    """
    Скачать Pinterest dataset
    https://data.mendeley.com/datasets/fs4k2zc5j5/3

    :param path:
    :return:
    """
    logging.info("Getting Pinterest dataset...")
    folder = join(path, "pinterest")
    mkdir(folder)
    logging.info("[1/9]: Downloading images...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/01b599d6-4c60-46c7-9d6a-3fb7cb41fc71/images.rar?dl=1"
    download_dataset(url, join(folder, "images.rar"))

    logging.info("[2/9]: Downloading train images labels...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/cc4e3cb0-f2cf-4f4a-9395-a8215dc68b45/imag_train.txt?dl=1"
    download_url(url, join(folder, "imag_train.txt"))

    logging.info("[3/9]: Downloading val images labels...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/5a5ba856-9f71-4398-89bf-6942554c3d5c/imag_val.txt?dl=1"
    download_url(url, join(folder, "imag_val.txt"))

    logging.info("[4/9]: Downloading test images labels...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/e7ccc924-fb3e-476d-8e47-f96e96341263/imag_test.txt?dl=1"
    download_url(url, join(folder, "imag_test.txt"))

    logging.info("[5/9]: Downloading train text labels...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/a74646ce-56ef-407e-bfc3-1ad6198273db/text_train.txt?dl=1"
    download_url(url, join(folder, "text_train.txt"))

    logging.info("[6/9]: Downloading val text labels...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/b4657aa6-481f-4782-a3f9-1156f0451109/text_val.txt?dl=1"
    download_url(url, join(folder, "text_val.txt"))

    logging.info("[7/9]: Downloading test text labels...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/1d3d65ff-4de3-40ef-a54d-c2a19546c642/text_test.txt?dl=1"
    download_url(url, join(folder, "text_test.txt"))

    logging.info("[8/9]: Downloading train users...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/85b80304-358e-4e09-880c-939a480d684b/train_users.txt?dl=1"
    download_url(url, join(folder, "train_users.txt"))

    logging.info("[9/9]: Downloading val/test users...")
    url = "https://data.mendeley.com/datasets/fs4k2zc5j5/3/files/e90f3af5-9977-40cb-aaed-2491f8376982/val_test_users.txt?dl=1"
    download_url(url, join(folder, "val_test_users.txt"))
