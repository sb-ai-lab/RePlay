import sys
import logging
import requests

from sponge_bob_magic.data_loader.archives import extract, delete


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
        delete(archive_name)
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
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            chunk_size = max(int(total / 1000), 1024 * 1024)
            for data in response.iter_content(chunk_size=chunk_size):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                sys.stdout.flush()

    sys.stdout.write('\n')
