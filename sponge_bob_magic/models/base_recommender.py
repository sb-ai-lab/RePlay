"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from abc import ABC, abstractmethod
from shutil import rmtree
from subprocess import check_output
from typing import Any, Dict, Optional, Set

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import DEFAULT_CONTEXT


class BaseRecommender(ABC):
    """ Базовый класс-рекомендатель. """
    model: Any = None
    to_overwrite_files: bool = True

    def __init__(self, spark: SparkSession, **kwargs):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param spark: инициализированная спарк-сессия
        :param kwargs: параметры для модели
        """
        self.spark = spark

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Устанавливает параметры рекоммендера.

        :param params: словарь, ключ - название параметра,
            значение - значение параметра
        :return:
        """
        valid_params = self.get_params()

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    "Неправильный параметр для данного рекоммендера "
                    f"{param}. "
                    "Проверьте список параметров, "
                    f"которые принимает рекоммендер: {valid_params}")

            setattr(self, param, value)

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """
        Возвращает параметры рекоммендера в виде словаря.

        :return: словарь параметров, ключ - название параметра,
            значение - значение параметра
        """

    @staticmethod
    def _check_dataframe(dataframe: Optional[DataFrame],
                         required_columns: Set[str],
                         optional_columns: Set[str]) -> None:
        if dataframe is None:
            raise ValueError("Датафрейм есть None")

        # чекаем, что датафрейм не пустой
        if len(dataframe.head(1)) == 0:
            raise ValueError("Датафрейм пустой")

        df_columns = dataframe.columns

        # чекаем на нуллы
        for column in df_columns:
            if dataframe.where(sf.col(column).isNull()).count() > 0:
                raise ValueError(f"В колонке '{column}' есть значения NULL")

        # чекаем колонки
        if not required_columns.issubset(df_columns):
            raise ValueError(
                f"В датафрейме нет обязательных колонок ({required_columns})")

        wrong_columns = (set(df_columns)
                         .difference(required_columns)
                         .difference(optional_columns))
        if len(wrong_columns) > 0:
            raise ValueError(
                f"В датафрейме есть лишние колонки: {wrong_columns}")

    @staticmethod
    def _check_feature_dataframe(features: Optional[DataFrame],
                                 required_columns: Set[str],
                                 optional_columns: Set[str]) -> None:
        if features is None:
            return

        columns = set(features.columns).difference(required_columns)
        if len(columns) == 0:
            raise ValueError("В датафрейме features нет колонок с фичами")

        BaseRecommender._check_dataframe(
            features,
            required_columns=required_columns.union(columns),
            optional_columns=optional_columns
        )

    @staticmethod
    def _check_input_dataframes(log: DataFrame,
                                user_features: DataFrame,
                                item_features: DataFrame) -> None:
        BaseRecommender._check_dataframe(
            log,
            required_columns={"item_id", "user_id", "timestamp", "relevance",
                              "context"},
            optional_columns=set()
        )
        BaseRecommender._check_feature_dataframe(
            user_features,
            optional_columns=set(),
            required_columns={"user_id", "timestamp"}
        )
        BaseRecommender._check_feature_dataframe(
            item_features,
            optional_columns=set(),
            required_columns={"item_id", "timestamp"}
        )

    def fit(self, log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            path: Optional[str] = None) -> None:
        """
        Обучает модель на логе и признаках пользователей и объектов.

        :param path: путь к директории, в которой сохраняются промежуточные
            резльтаты в виде parquet-файлов; если None, делаются checkpoints
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :return:
        """
        logging.debug("Проверка датафреймов")
        self._check_input_dataframes(log, user_features, item_features)

        logging.debug("Предварительная стадия обучения (pre-fit)")
        self._pre_fit(log, user_features, item_features, path)

        logging.debug("Основная стадия обучения (fit)")
        self._fit_partial(log, user_features, item_features, path)

    @abstractmethod
    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 path: Optional[str] = None) -> None:
        """
        Метод-helper для обучения модели, в которой параметры не используются.
        Нужен для того, чтобы вынести вычисление трудоемких агрегатов
        в отдельный метод, который по возможности будет вызываться один раз.
        Должен быть имплементирован наследниками.

        :param path: путь к директории, в которой сохраняются промежуточные
            резльтаты в виде parquet-файлов; если None, делаются checkpoints
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :return:
        """

    @abstractmethod
    def _fit_partial(self, log: DataFrame,
                     user_features: Optional[DataFrame],
                     item_features: Optional[DataFrame],
                     path: Optional[str] = None) -> None:
        """
        Метод-helper для обучения модели, в которой используются параметры.
        Должен быть имплементирован наследниками.

        :param path: путь к директории, в которой сохраняются промежуточные
            резльтаты в виде parquet-файлов; если None, делаются checkpoints
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :return:
        """

    def predict(
            self,
            k: int,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            context: Optional[str],
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            to_filter_seen_items: bool = True,
            path: Optional[str] = None,
            batch_size: Optional[int] = None
    ) -> DataFrame:
        """
        Выдача рекомендаций для пользователей.

        :param path: путь к директории, в которой сохраняются рекомендации;
            если None, делается checkpoint рекомендаций
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой `[user_id]`;
            если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой `[item_id]`;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в relevance в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендации;
            если None, контекст не будет использоваться
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param to_filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :param batch_size: скольким пользователям одновременно выдавать
            рекомендации. Если не задано, то всем сразу.
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        logging.debug("Проверка датафреймов")
        self._check_input_dataframes(log, user_features, item_features)
        if users is None:
            logging.debug("Выделение дефолтных юзеров")
            users = log.select("user_id").distinct()
        if items is None:
            logging.debug("Выделение дефолтных айтемов")
            items = log.select("item_id").distinct()
        if context is None:
            context = DEFAULT_CONTEXT
        num_items = items.count()
        num_users = users.count()
        if num_items < k:
            raise ValueError(
                "Значение k больше, чем множество объектов; "
                f"k = {k}, number of items = {num_items}")
        if batch_size is None or batch_size >= num_users:
            recs = self._predict(
                k, users, items, context, log,
                user_features, item_features, to_filter_seen_items, path
            )
        else:
            if path is None:
                raise ValueError("""При выдаче рекомендаций батчами всегда
 нужно задавать путь для сохранения рекомендаций!""")
            recs = self._batch_predict(
                k, users, items, context, log,
                user_features, item_features, batch_size, num_users,
                path, to_filter_seen_items
            )
        return recs

    def _batch_predict(
            self,
            k: int,
            users: DataFrame,
            items: DataFrame,
            context: str,
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            batch_size: int,
            num_users: int,
            path: str,
            to_filter_seen_items: bool = True
    ) -> DataFrame:
        """
        Разбитие списка пользователей на батчи и выдача рекомендаций по ним.

        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации, спарк-датафрейм с колонкой `[user_id]`;
            если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            спарк-датафрейм с колонкой `[item_id]`;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в relevance в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендации;
            если None, контекст не будет использоваться
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param batch_size: сколько пользователей включать в батч
        :param num_users: общее количество пользователей
        :param to_filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :param path: путь к директории, в которой сохраняются рекомендации
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        batches_path = os.path.join(path, "batches")
        users.repartition(
            num_users // batch_size
        ).write.mode("overwrite").parquet(batches_path)
        rmtree(os.path.join(path, "batch_recs"), ignore_errors=True)
        for partition_num, partition_filename in enumerate(
                os.listdir(batches_path)
        ):
            if os.path.splitext(partition_filename)[1] == ".parquet":
                users_batch = self.spark.read.parquet(
                    os.path.join(batches_path, partition_filename)
                )
                self._predict(
                    k=k,
                    users=users_batch,
                    items=items,
                    log=log,
                    context=context if context else DEFAULT_CONTEXT,
                    user_features=user_features,
                    item_features=item_features,
                    to_filter_seen_items=to_filter_seen_items,
                    path=os.path.join(
                        path, f"batch_recs/batch{partition_num}"
                    )
                )
        parquet_wildcard = os.path.join(
            path, "batch_recs/batch*/recs.parquet/*.parquet"
        )
        recs_dir = os.path.join(path, "recs.parquet")
        rmtree(recs_dir, ignore_errors=True)
        os.mkdir(recs_dir)
        logging.debug(
            check_output([f"mv {parquet_wildcard} {recs_dir}"], shell=True)
        )
        return self.spark.read.parquet(recs_dir)

    @abstractmethod
    def _predict(self,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 context: str,
                 log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True,
                 path: Optional[str] = None) -> DataFrame:
        """
        Метод-helper для получения рекомендаций.
        Должен быть имплементирован наследниками.

        :param path: путь к директории, в которой сохраняются рекомендации;
            если None, делается checkpoint рекомендаций
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то вызывается ошибка
        :param items: список объектов, которые необходимо рекомендовать;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендоции;
            если None, контекст не будет использоваться
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param to_filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """

    def fit_predict(
            self,
            k: int,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            context: Optional[str],
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            to_filter_seen_items: bool = True,
            path: Optional[str] = None,
            batch_size: int = None
    ) -> DataFrame:
        """
        Обучает модель и выдает рекомендации.

        :param path: путь к директории, в которой сохраняются рекомендации;
            если None, делается checkpoint рекомендаций
        :param k: количество рекомендаций для каждого пользователя;
            должно быть не больше, чем количество объектов в `items`
        :param users: список пользователей, для которых необходимо получить
            рекомендации; если None, выбираются все пользователи из лога;
            если в этом списке есть пользователи, про которых модель ничего
            не знает, то поднмиается исключение
        :param items: список объектов, которые необходимо рекомендовать;
            если None, выбираются все объекты из лога;
            если в этом списке есть объекты, про которых модель ничего
            не знает, то в рекомендациях к ним будет стоять 0
        :param context: контекст, в котором нужно получить рекомендоции;
            если None, контекст не будет использоваться
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            `[user_id , timestamp]` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            `[item_id , timestamp]` и колонки с признаками
        :param to_filter_seen_items: если True, из рекомендаций каждому
            пользователю удаляются виденные им объекты на основе лога
        :param batch_size: скольким пользователям выдавать рекомендации
            одновременно. Если не задано, то всем сразу.
        :return: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        self.fit(log, user_features, item_features, path)
        return self.predict(k, users, items,
                            context, log,
                            user_features, item_features,
                            to_filter_seen_items,
                            path, batch_size)

    @staticmethod
    def _filter_seen_recs(recs: DataFrame, log: DataFrame) -> DataFrame:
        """
        Преобразует рекомендации, заменяя для каждого пользователя
        relevance уже виденных им объекты (на основе лога) на -1.

        :param recs: рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками
            `[user_id , item_id , timestamp , context , relevance]`
        :return: измененные рекомендации, спарк-датафрейм с колонками
            `[user_id , item_id , context , relevance]`
        """
        user_item_log = (log
                         .select("item_id", "user_id")
                         .withColumn("in_log", sf.lit(True)))
        recs = (recs
                .join(user_item_log, on=["item_id", "user_id"], how="left"))
        recs = (recs
                .withColumn("relevance",
                            sf.when(recs["in_log"], -1)
                            .otherwise(recs["relevance"]))
                .drop("in_log"))
        return recs
