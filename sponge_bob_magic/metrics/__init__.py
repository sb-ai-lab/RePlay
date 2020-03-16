"""
Для рассчета большинства метрик требуется таблица с рекомендациями и таблица с реальными значениями -- списком айтемов,
с которыми провзаимодействовал пользователь.

Все метрики рассчитываются для первых ``K`` объектов в рекомендации.
Поддерживается возможность рассчета метрик сразу по нескольким ``K``,
в таком случае будет возвращен словарь с результатами, а не число.

Если реализованных метрик недостаточно, библиотека поддерживает возможность
:ref:`добавления своих метрик <new-metric>`.
"""
from sponge_bob_magic.metrics.base_metric import Metric
from sponge_bob_magic.metrics.coverage import Coverage
from sponge_bob_magic.metrics.hitrate import HitRate
from sponge_bob_magic.metrics.map import MAP
from sponge_bob_magic.metrics.mrr import MRR
from sponge_bob_magic.metrics.ndcg import NDCG
from sponge_bob_magic.metrics.precision import Precision
from sponge_bob_magic.metrics.recall import Recall
from sponge_bob_magic.metrics.surprisal import Surprisal
from sponge_bob_magic.metrics.unexpectedness import Unexpectedness
