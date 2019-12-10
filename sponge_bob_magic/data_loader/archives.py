import os
from os.path import splitext
from zipfile import ZipFile


def extract(archive_name: str):
    """
    Извлечь содержимое архива и положить его в папку, если там несколько файлов.

    :param archive_name: путь до архива
    :return: None
    """
    archive = ZipFile(archive_name)
    contents = archive.namelist()

    if have_folder(contents):
        name = None
    else:
        name = remove_extension(archive_name)
        os.mkdir(name)

    archive.extractall(path=name)


def delete(archive_name: str):
    """
    Удалить архив, чтобы не мешался.

    :param archive_name: путь до архива
    :return: None
    """
    if os.path.exists(archive_name):
        os.remove(archive_name)


def have_folder(contents: list):
    """
    Проверить, что архив содержит папку, содержащую все остальное.

    :param contents: списое содержимого из .namelist()
    :return: None
    """
    return is_dir(contents[0])


def is_dir(s: str):
    """
    Проверка того, что строка обозначает папку.

    :param s: строка
    :return: None
    """
    last_char = s[-1]
    return True if last_char == '/' or last_char == '\\' else False


def remove_extension(file: str):
    """
    Получить имя файла без расширения.

    :param file: строка
    :return: None
    """
    return splitext(file)[0]
