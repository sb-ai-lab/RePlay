"""
Для рассчета большинства метрик требуется таблица с рекомендациями и таблица с реальными значениями -- списком айтемов,
с которыми провзаимодействовал пользователь.

Все метрики рассчитываются для первых ``K`` объектов в рекомендации.
Поддерживается возможность рассчета метрик сразу по нескольким ``K``,
в таком случае будет возвращен словарь с результатами, а не число.

Если реализованных метрик недостаточно, библиотека поддерживает возможность
:ref:`добавления своих метрик <new-metric>`.
"""
from replay.metrics.base_metric import Metric
from replay.metrics.coverage import Coverage
from replay.metrics.hitrate import HitRate
from replay.metrics.map import MAP
from replay.metrics.mrr import MRR
from replay.metrics.ndcg import NDCG
from replay.metrics.precision import Precision
from replay.metrics.recall import Recall
from replay.metrics.rocauc import RocAuc
from replay.metrics.surprisal import Surprisal
from replay.metrics.unexpectedness import Unexpectedness
