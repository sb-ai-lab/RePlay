from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from ignite.contrib.handlers import LRScheduler
from ignite.engine import Engine, Events
from ignite.handlers import (
    EarlyStopping,
    ModelCheckpoint,
    global_step_from_engine,
)
from ignite.metrics import Loss, RunningAverage
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from torch import nn
from torch.optim import optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from sponge_bob_magic.models import Recommender


class TorchRecommender(Recommender):
    """ Базовый класс-рекомендатель для нейросетевой модели. """

    device: torch.device

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        items_pd = (
            self.item_indexer.transform(items).toPandas()["item_idx"].values
        )
        items_count = self.items_count
        model = self.model.cpu()
        agg_fn = self._predict_by_user

        @sf.pandas_udf(
            st.StructType(
                [
                    st.StructField("user_idx", st.LongType(), True),
                    st.StructField("item_idx", st.LongType(), True),
                    st.StructField("relevance", st.FloatType(), True),
                ]
            ),
            sf.PandasUDFType.GROUPED_MAP,
        )
        def grouped_map(pandas_df):
            return agg_fn(pandas_df, model, items_pd, k, items_count)[
                ["user_idx", "item_idx", "relevance"]
            ]

        self.logger.debug("Предсказание модели")
        recs = self.item_indexer.transform(
            self.user_indexer.transform(
                users.join(log, how="left", on="user_id")
            )
        )
        recs = (
            recs.selectExpr(
                "CAST(user_idx AS INT) AS user_idx",
                "CAST(item_idx AS INT) AS item_idx",
            )
            .groupby("user_idx")
            .apply(grouped_map)
        )
        recs = self.inv_item_indexer.transform(
            self.inv_user_indexer.transform(recs)
        ).drop("item_idx", "user_idx")

        recs = self.min_max_scale_column(recs, "relevance")
        return recs

    @staticmethod
    @abstractmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.array,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
        """
        Расчёт значения метрики для каждого пользователя

        :param pandas_df: DataFrame, содержащий индексы просмотренных айтемов
            по каждому пользователю -- pandas-датафрейм вида
            ``[user_idx, item_idx]``
        :param model: обученная модель
        :param items_np: список допустимых для рекомендаций объектов
        :param k: количество рекомендаций
        :param item_count: общее количество объектов в рекомендателе
        :return: DataFrame c рассчитанными релевантностями --
            pandas-датафрейм вида ``[user_idx , item_idx , relevance]``
        """

    @staticmethod
    def min_max_scale_column(dataframe: DataFrame, column: str) -> DataFrame:
        """
        Отнормировать колонку датафрейма.
        Применяет классическую форму нормализации с минимумом и максимумом:
        new_value_i = (value_i - min) / (max - min).

        :param dataframe: спарк-датафрейм
        :param column: имя колонки, которую надо нормализовать
        :return: исходный датафрейм с измененной колонкой
        """
        unlist = sf.udf(lambda x: float(list(x)[0]), st.DoubleType())
        assembler = VectorAssembler(
            inputCols=[column], outputCol=f"{column}_Vect"
        )
        scaler = MinMaxScaler(
            inputCol=f"{column}_Vect", outputCol=f"{column}_Scaled"
        )
        pipeline = Pipeline(stages=[assembler, scaler])
        dataframe = (
            pipeline.fit(dataframe)
            .transform(dataframe)
            .withColumn(column, unlist(f"{column}_Scaled"))
            .drop(f"{column}_Vect", f"{column}_Scaled")
        )

        return dataframe

    def load_model(self, path: str) -> None:
        """
        Загрузка весов модели из файла

        :param path: путь к файлу, откуда загружать
        :return:
        """
        self.logger.debug("-- Загрузка модели из файла")
        self.model.load_state_dict(torch.load(path))

    def _create_trainer_evaluator(
        self,
        opt: optimizer,
        valid_data_loader: DataLoader,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        early_stopping_patience: Optional[int] = None,
        checkpoint_number: Optional[int] = None,
    ) -> (Engine, Engine):
        """
        Метод, возвращающий trainer, evaluator для обучения нейронной сети.

        :param opt: Оптимимайзер
        :param valid_data_loader: Загрузчик данных для валидации
        :param scheduler: Расписания для уменьшения шага обучения
        :param early_stopping_patience: количество эпох для ранней остановки
        :param early_stopping_patience: количество лучших чекпойнтов
        :return: trainer, evaluator
        """
        self.model.to(self.device)

        def _run_train_step(engine, batch):
            self.model.train()
            opt.zero_grad()
            model_result = self._batch_pass(batch, self.model)
            y_pred, y_true = model_result[:2]
            if len(model_result) == 2:
                loss = self._loss(y_pred, y_true)
            else:
                loss = self._loss(y_pred, y_true, **model_result[2])
            loss.backward()
            opt.step()
            return loss.item()

        def _run_val_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                return self._batch_pass(batch, self.model)

        torch_trainer = Engine(_run_train_step)
        torch_evaluator = Engine(_run_val_step)

        avg_output = RunningAverage(output_transform=lambda x: x)
        avg_output.attach(torch_trainer, "loss")
        Loss(self._loss).attach(torch_evaluator, "loss")

        @torch_trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(trainer):
            self.logger.debug(
                "Epoch[{}] current loss: {:.5f}".format(
                    trainer.state.epoch, trainer.state.metrics["loss"]
                )
            )

        @torch_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            torch_evaluator.run(valid_data_loader)
            metrics = torch_evaluator.state.metrics
            self.logger.debug(
                "Epoch[{}] validation average loss: {:.5f}".format(
                    trainer.state.epoch, metrics["loss"]
                )
            )

        def score_function(engine):
            return -engine.state.metrics["loss"]

        if early_stopping_patience:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                score_function=score_function,
                trainer=torch_trainer,
            )
            torch_evaluator.add_event_handler(Events.COMPLETED, early_stopping)
        if checkpoint_number:
            checkpoint = ModelCheckpoint(
                self.spark.conf.get("spark.local.dir"),
                create_dir=True,
                require_empty=False,
                n_saved=checkpoint_number,
                score_function=score_function,
                score_name="loss",
                filename_prefix="best",
                global_step_transform=global_step_from_engine(torch_trainer),
            )

            torch_evaluator.add_event_handler(
                Events.EPOCH_COMPLETED,
                checkpoint,
                {type(self).__name__.lower(): self.model},
            )

            @torch_trainer.on(Events.COMPLETED)
            def load_best_model(engine):
                self.load_model(checkpoint.last_checkpoint)

        if scheduler:
            if isinstance(scheduler, _LRScheduler):
                torch_trainer.add_event_handler(
                    Events.EPOCH_COMPLETED, LRScheduler(scheduler)
                )
            else:

                @torch_evaluator.on(Events.EPOCH_COMPLETED)
                def reduct_step(engine):
                    scheduler.step(engine.state.metrics["loss"])

        return torch_trainer, torch_evaluator

    @abstractmethod
    def _batch_pass(
        self, batch, model
    ) -> (torch.Tensor, torch.Tensor, Union[None, Dict[str, Any]]):
        """
        Метод, возвращающий результат применения модели к батчу.
        Должен быть имплементирован наследниками.

        :param batch: батч с данными
        :param model: нейросетевая модель
        :return: y_pred, y_true, а также словарь дополнительных параметров,
        необходимых для расчета функции потерь
        """

    @abstractmethod
    def _loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Метод, возвращающий значение функции потерь.
        Должен быть имплементирован наследниками.

        :param y_pred: Результат, который вернула нейросеть
        :param y_true: Ожидаемый результат
        :param *args: Прочие аргументы необходимые для расчета loss
        :return: Тензор размера 1 на 1
        """
