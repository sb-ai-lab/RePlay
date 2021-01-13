"""
Для расчета большинства метрик требуется таблица с рекомендациями и
таблица с реальными значениями -- списком объектов,
с которыми взаимодействовал пользователь.

- recommendations (Union[pandas.DataFrame, spark.DataFrame]):
    выдача рекомендательной системы, спарк-датафрейм вида ``[user_id, item_id, relevance]``
- ground_truth (Union[pandas.DataFrame, spark.DataFrame]):
    реальный лог действий пользователей, спарк-датафрейм вида ``[user_id, item_id, timestamp, relevance]``

Все метрики рассчитываются для первых ``K`` объектов в рекомендации.
Поддерживается возможность расчета метрик сразу по нескольким ``K``,
в таком случае будет возвращен словарь с результатами, а не число.

- k (Union[Iterable[int], int]):
    список индексов, показывающий сколько объектов брать из топа рекомендованных

По умолчанию возвращается среднее значение по пользователям,
но можно воспользоваться методом ``metric.median``.

Кроме того, можно получить нижнюю границу доверительного интервала для заданного ``alpha``
с помощью метода ``conf_interval``.

Метрики разнообразия оценивают рекомендации, не сравнивая их с ``ground_truth``.
Такие метрики инициализируются дополнительными параметрами, а при вызове передается только список рекомендаций
и значение k.

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
