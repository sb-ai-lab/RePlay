"""
Сплиттеры реализуют стратегии разделения данных на тренировочную и тестовую выборки.
Использование одинаковых сплитов полезно для сравнения моделей между собой.
"""

from sponge_bob_magic.splitters.log_splitter import *
from sponge_bob_magic.splitters.user_log_splitter import RandomUserSplitter, TimeUserSplitter
from sponge_bob_magic.splitters.base_splitter import *