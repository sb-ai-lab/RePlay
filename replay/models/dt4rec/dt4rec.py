from typing import Optional

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from replay.models.base_rec import Recommender
from replay.utils.spark_utils import convert2spark

from .gpt1 import GPT, GPTConfig
from .trainer import Trainer, TrainerConfig
from .utils import (
    Collator,
    StateActionReturnDataset,
    ValidateDataset,
    WarmUpScheduler,
    create_dataset,
    matrix2df,
    set_seed,
)


class DT4Rec(Recommender):
    optimizer = None
    train_batch_size = 128
    val_batch_size = 128
    lr_scheduler = None

    def __init__(
        self,
        item_num,
        user_num,
        seed=123,
        trajectory_len=30,
        epochs=1,
        batch_size=64,
    ):
        self.item_num = item_num
        self.user_num = user_num
        self.seed = seed
        self.trajectory_len = trajectory_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.tconf: TrainerConfig = TrainerConfig(epochs=epochs)
        self.mconf: GPTConfig = GPTConfig(
            user_num=user_num,
            item_num=item_num,
            vocab_size=self.item_num + 1,
            block_size=self.trajectory_len * 3,
            max_timestep=self.item_num,
        )
        self.model: GPT
        set_seed(self.seed)

    def _update_mconf(self, **kwargs):
        self.mconf.update(**kwargs)

    def _update_tconf(self, **kwargs):
        self.tconf.update(**kwargs)

    def _make_prediction_dataloader(self, users, items, max_context_len=30):
        val_dataset = ValidateDataset(
            self.user_trajectory,
            max_context_len=max_context_len - 1,
            val_items=users,
            val_users=items,
        )

        val_dataloader = DataLoader(
            val_dataset,
            pin_memory=True,
            batch_size=self.val_batch_size,
            collate_fn=Collator(self.item_num),
        )

        return val_dataloader

    def train(
        self,
        log,
        val_users=None,
        val_items=None,
        experiment=None,
    ):
        assert (val_users is None) == (val_items is None) == (experiment is None)
        with_validate = experiment is not None
        df = log.toPandas()[["user_idx", "item_idx", "relevance", "timestamp"]]
        self.user_trajectory = create_dataset(df, user_num=self.user_num, item_pad=self.item_num)

        train_dataset = StateActionReturnDataset(self.user_trajectory, self.trajectory_len)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.train_batch_size,
            collate_fn=Collator(self.item_num),
        )

        if with_validate:
            val_dataloader = self._make_prediction_dataloader(val_users, val_items, max_context_len=self.trajectory_len)
        else:
            val_dataloader = None

        self.model = GPT(self.mconf)

        optimizer = torch.optim.AdamW(
            self.model.configure_optimizers(),
            lr=3e-4,
            betas=(0.9, 0.95),
        )
        lr_scheduler = WarmUpScheduler(optimizer, dim_embed=768, warmup_steps=4000)

        self.tconf.update(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.trainer = Trainer(
            self.model,
            train_dataloader,
            self.tconf,
            val_dataloader,
            experiment,
        )
        self.trainer.train()

    def _init_args(self):
        pass

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.train(log)

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
        items_consider_in_pred = items.toPandas()["item_idx"].values
        users_consider_in_pred = users.toPandas()["user_idx"].values
        ans = self._predict_helper(users_consider_in_pred, items_consider_in_pred)
        return convert2spark(ans)

    def _predict_helper(self, users, items, max_context_len=30):
        predict_dataloader = self._make_prediction_dataloader(users, items, max_context_len)
        self.model.eval()
        ans_df = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
        with torch.no_grad():
            for batch in tqdm(predict_dataloader):
                states, actions, rtgs, timesteps, users = self.trainer._move_batch(batch)
                logits = self.model(states, actions, rtgs, timesteps, users)
                items_relevances = logits[:, -1, :][:, items]
                ans_df = ans_df.append(matrix2df(items_relevances, users.squeeze(), items))

        return ans_df
