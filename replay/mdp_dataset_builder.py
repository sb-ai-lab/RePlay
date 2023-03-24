import logging
import timeit

import numpy as np
from d3rlpy.dataset import MDPDataset
from pyspark.sql import DataFrame, functions as sf, Window

timer = timeit.default_timer


class MdpDatasetBuilder:
    logger: logging.Logger
    top_k: int
    action_randomization_scale: float

    def __init__(self, top_k: int, action_randomization_scale: float = 1e-3):
        self.logger = logging.getLogger("replay")
        self.top_k = top_k
        # cannot set zero scale as then d3rlpy will treat transitions as discrete
        assert action_randomization_scale > 0
        self.action_randomization_scale = action_randomization_scale

    def build(self, log: DataFrame) -> MDPDataset:
        start_time = timer()
        # reward top-K watched movies with 1, the others - with 0
        reward_condition = sf.row_number().over(
            Window
            .partitionBy('user_idx')
            .orderBy([sf.desc('relevance'), sf.desc('timestamp')])
        ) <= self.top_k

        # every user has his own episode (the latest item is defined as terminal)
        terminal_condition = sf.row_number().over(
            Window
            .partitionBy('user_idx')
            .orderBy(sf.desc('timestamp'))
        ) == 1

        user_logs = (
            log
            .withColumn("reward", sf.when(reward_condition, sf.lit(1)).otherwise(sf.lit(0)))
            .withColumn("terminal", sf.when(terminal_condition, sf.lit(1)).otherwise(sf.lit(0)))
            .withColumn(
                "action",
                sf.col("relevance").cast("float") + sf.randn() * self.action_randomization_scale
            )
            .orderBy(['user_idx', 'timestamp'], ascending=True)
            .select(['user_idx', 'item_idx', 'action', 'reward', 'terminal'])
            .toPandas()
        )
        train_dataset = MDPDataset(
            observations=np.array(user_logs[['user_idx', 'item_idx']]),
            actions=user_logs['action'].to_numpy()[:, None],
            rewards=user_logs['reward'].to_numpy(),
            terminals=user_logs['terminal'].to_numpy()
        )

        prepare_time = timer() - start_time
        self.logger.info(f'-- Building MDP dataset took {prepare_time:.2f} seconds')
        return train_dataset

    def init_args(self):
        return dict(
            top_k=self.top_k,
            action_randomization_scale=self.action_randomization_scale
        )
