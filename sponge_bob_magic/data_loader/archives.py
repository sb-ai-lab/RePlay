"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from os.path import splitext
from zipfile import ZipFile


def extract(archive_name: str) -> None:
    """
    Извлечь содержимое архива и положить его в папку, если там несколько
    файлов.

    :param archive_name: путь до архива
    :return:
    """
    archive = ZipFile(archive_name)
    if contains_dir(archive):
        name = os.path.dirname(archive_name)
    else:
        name = remove_extension(archive_name)
        os.mkdir(name)
    archive.extractall(path=name)


def rm_if_exists(filename: str) -> None:
    """
    Удалить файл, если он существует, а если не существует, не бросать
    исключение.

    :param filename: имя файла
    :return:
    """
    if os.path.exists(filename):
        os.remove(filename)


def contains_dir(zip_file: ZipFile) -> bool:
    """
    Проверить, запакована ли в архив папка или просто набор файлов.

    :param zip_file: zip-архив
    :return: является ли первый элемент содержимого арихва папкой
    """
    contents = zip_file.infolist()
    return contents[0].is_dir()


def remove_extension(file: str) -> str:
    """
    Получить имя файла без _последнего_ расширения.

    :param file: строка
    :return: archive.tar.gz -> archive.tar
    """
    return splitext(file)[0]
