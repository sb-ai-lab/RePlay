"""Данный модуль осуществляет загрузку и предоставляет доступ к публичным датасетам.
Датасет скачивается и парсится, данные доступны как атрибуты объекта в виде ``pandas.DataFrame``

По умолчанию датасеты скачиваются и считываются из папки ``/Users/username/sb_magic_data/``.
Задать дефолтную папку можно с помощью переменной окружения ``KRUSTY_KRABS``.
Поведение по умолчанию всегда можно переопределить с помощью параметра ``path`` при инициализации датасета.
"""
from sponge_bob_magic.datasets.movielens import MovieLens
from sponge_bob_magic.datasets.msd import MillionSongDataset
from sponge_bob_magic.datasets.netflix import Netflix
