"""Данный модуль осуществляет загрузку и предоставляет доступ к публичным датасетам.
Датасет скачивается и парсится, данные доступны как атрибуты объекта в виде ``pandas.DataFrame``"""
from sponge_bob_magic.datasets.movielens import MovieLens
from sponge_bob_magic.datasets.msd import MillionSongDataset
from sponge_bob_magic.datasets.netflix import Netflix
