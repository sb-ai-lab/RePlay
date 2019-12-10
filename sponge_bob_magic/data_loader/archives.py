import os
from zipfile import ZipFile


def extract(archive_name):

    archive = ZipFile(archive_name)
    contents = archive.namelist()

    if have_folder(contents):
        name = None
    else:
        name = remove_extension(archive_name)
        os.mkdir(name)

    archive.extractall(path=name)


def delete(archive_name):
    if os.path.exists(archive_name):
        os.remove(archive_name)


def have_folder(contents):
    return is_dir(contents[0])


def is_dir(s):
    last_char = s[-1]
    return True if last_char == '/' or last_char == '\\' else False


def remove_extension(file):
    return file.split('.')[:-1]
