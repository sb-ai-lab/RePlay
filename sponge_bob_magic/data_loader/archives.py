"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
import tarfile
from os.path import splitext
from zipfile import ZipFile
from tarfile import TarFile


def extract(archive_name: str, manage_folder: bool = True) -> None:
    """
    Извлечь содержимое архива и положить его в папку, если там несколько
    файлов.

    :param archive_name: путь до архива
    :param manage_folder: проверить наличие корневой папки в архиве:
        если она есть, не создавать дополнительную папку,
        если в архиве лежат просто файлы, положить их в папку.
        Значение параметра False означает распаковку "как есть".
    :return:
    """
    if archive_name.endswith(".zip"):
        archive = ZipFile(archive_name)
    elif archive_name.endswith(".tar.gz"):
        archive = tarfile.open(archive_name, "r:gz")
    elif archive_name.endswith(".tar"):
        archive = tarfile.open(archive_name, "r:")
    elif archive_name.endswith(".bz2"):
        archive = tarfile.open(archive_name, "r:bz2")
    else:
        raise NotImplementedError(f"Can\'t extract {archive_name}")

    name = os.path.dirname(archive_name)
    if manage_folder and not contains_dir(archive):
        name = remove_extension(archive_name)
        os.mkdir(name)

    archive.extractall(path=name)
    archive.close()


def rm_if_exists(filepath: str) -> None:
    """
    Удалить файл, если он существует, а если не существует, не бросать
    исключение.

    :param filepath: путь до файла
    :return:
    """
    if os.path.exists(filepath):
        os.remove(filepath)


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
