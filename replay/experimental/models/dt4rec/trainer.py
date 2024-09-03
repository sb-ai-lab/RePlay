import logging

import pandas as pd
from tqdm import tqdm

from replay.utils import TORCH_AVAILABLE

from .utils import matrix2df

if TORCH_AVAILABLE:
    import torch
    from torch.nn import functional as func


logger = logging.getLogger(__name__)


class TrainerConfig:
    """
    Config holder for trainer
    """

    epochs = 1
    lr_scheduler = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """
        Arguments setter
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:
    """
    Trainer for DT4Rec
    """

    grad_norm_clip = 1.0

    def __init__(
        self,
        model,
        train_dataloader,
        tconf,
        val_dataloader=None,
        experiment=None,
        use_cuda=True,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = tconf.optimizer
        self.epochs = tconf.epochs
        self.lr_scheduler = tconf.lr_scheduler
        assert (val_dataloader is None) == (experiment is None)
        self.val_dataloader = val_dataloader
        self.experiment = experiment

        # take over whatever gpus are on the system
        self.device = "cpu"
        if use_cuda and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def _move_batch(self, batch):
        return [elem.to(self.device) for elem in batch]

    def _train_epoch(self, epoch):
        self.model.train()

        losses = []
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
        )

        for iter_, batch in pbar:
            # place data on the correct device
            states, actions, rtgs, timesteps, users = self._move_batch(batch)
            targets = actions

            # forward the model
            logits = self.model(states, actions, rtgs, timesteps, users)

            loss = func.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).mean()
            losses.append(loss.item())

            # backprop and update the parametersx
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # report progress
            if self.lr_scheduler is not None:
                current_lr = self.lr_scheduler.get_lr()
            else:
                current_lr = self.optimizer.param_groups[-1]["lr"]
            pbar.set_description(f"epoch {epoch+1} iter {iter_}: train loss {loss.item():.5f}, lr {current_lr}")

    def _evaluation_epoch(self, epoch):
        self.model.eval()
        ans_df = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
        val_items = self.val_dataloader.dataset.val_items
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                states, actions, rtgs, timesteps, users = self._move_batch(batch)
                logits = self.model(states, actions, rtgs, timesteps, users)
                items_relevances = logits[:, -1, :][:, val_items]
                ans_df = ans_df.append(matrix2df(items_relevances, users.squeeze(), val_items))
            self.experiment.add_result(f"epoch: {epoch}", ans_df)
            self.experiment.results.to_csv("results.csv")

    def train(self):
        """
        Run training loop
        """
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            if self.experiment is not None:
                self._evaluation_epoch(epoch)
