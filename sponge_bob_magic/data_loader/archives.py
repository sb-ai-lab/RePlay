"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from os.path import splitext
from zipfile import ZipFile


def extract(archive_name: str):
    """
    Извлечь содержимое архива и положить его в папку, если там несколько
    файлов.

    :param archive_name: путь до архива
    :return: None
    """
    archive = ZipFile(archive_name)

    if contains_dir(archive):
        name = "."
    else:
        name = remove_extension(archive_name)
        os.mkdir(name)

    archive.extractall(path=name)


def safe_delete(filename: str):
    """
    Удалить архив (или любой файл), чтобы не мешался.

    :param filename: путь до файла
    :return: None
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


def remove_extension(file: str):
    """
    Получить имя файла без расширения.

    :param file: строка
    :return: None
    """
    return splitext(file)[0]
