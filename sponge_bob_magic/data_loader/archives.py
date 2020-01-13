"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
import tarfile
from os.path import splitext
from zipfile import ZipFile
from tarfile import TarFile


def extract(archive_name: str) -> None:
    """
    Извлечь содержимое архива и положить его в папку, если там несколько
    файлов.

    :param archive_name: путь до архива
    :return:
    """
    if archive_name.endswith(".zip"):
        archive = ZipFile(archive_name)
    elif archive_name.endswith(".tar.gz"):
        archive = tarfile.open(archive_name, "r:gz")
    elif archive_name.endswith(".tar"):
        archive = tarfile.open(archive_name, "r:")

    if contains_dir(archive):
        name = os.path.dirname(archive_name)
    else:
        name = remove_extension(archive_name)
        os.mkdir(name)

    archive.extractall(path=name)
    archive.close()


def rm_if_exists(filename: str) -> None:
    """
    Удалить файл, если он существует, а если не существует, не бросать
    исключение.

    :param filename: имя файла
    :return:
    """
    if os.path.exists(filename):
        os.remove(filename)


def contains_dir(archive) -> bool:
    """
    Проверить, запакована ли в архив папка или просто набор файлов.

    :param archive: файл архива -- .zip или .tar.gz
    :return: является ли первый элемент содержимого арихва папкой
    """
    if isinstance(archive, ZipFile):
        contents = archive.infolist()
        return contents[0].is_dir()
    elif isinstance(archive, TarFile):
        contents = archive.getmembers()
        return contents[0].isdir()
    else:
        raise TypeError(f"Unknown archive type: {type(archive)}")


def remove_extension(file: str) -> str:
    """
    Получить имя файла без _последнего_ расширения.

    :param file: строка
    :return: archive.tar.gz -> archive.tar
    """
    return splitext(file)[0]
