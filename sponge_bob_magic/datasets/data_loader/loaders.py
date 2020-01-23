"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from urllib.request import urlretrieve

from tqdm import tqdm

from sponge_bob_magic.datasets.data_loader.archives import extract, rm_if_exists


def download_dataset(url: str, destination_path: str, manage_folder: bool = True):
    """
    Скачать датасет из интернета.

    :param url: откуда
    :param destination_path: куда
    :param manage_folder: проверить наличие корневой папки в архиве:
        если она есть, не создавать дополнительную папку,
        если в архиве лежат просто файлы, положить их в папку.
        Значение параметра False означает распаковку "как есть".
    :return: None
    """
    download_url(url, destination_path)
    extract(destination_path, manage_folder)
    rm_if_exists(destination_path)


def download_url(url: str, filename: str):
    """
    Скачать что-то по ссылке.

    :param url: ссылка
    :param filename: куда сохранить
    :return: None
    """
    with tqdm(unit="B", unit_scale=True) as progress:
        def report(chunk, chunksize, total):
            progress.total = total
            progress.update(chunksize)
        return urlretrieve(url, filename, reporthook=report)
