"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from urllib.request import urlretrieve

from sponge_bob_magic.data_loader.archives import extract, safe_delete
from tqdm import tqdm


def download_dataset(link: str, archive_name: str):
    """
    Скачать датасет из интернета

    :param link: откуда
    :param archive_name: куда
    :return: None
    """

    try:
        download_url(link, archive_name)
        extract(archive_name)
        safe_delete(archive_name)
        logging.info('Done\n')

    except Exception as e:
        logging.exception(e)


def download_url(url: str, filename: str):
    """
    Скачать что-то по ссылке.

    :param url: ссылка
    :param filename: как сохранить
    :return: None
    """
    with tqdm(unit='B', unit_scale=True) as progress:
        def report(chunk, chunksize, total):
            progress.total = total
            progress.update(chunksize)
        return urlretrieve(url, filename, reporthook=report)
