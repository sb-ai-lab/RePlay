import math
from typing import Any, Optional, Tuple, Union, cast

import lightning as L
import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.models.nn.optimizer_utils import FatOptimizerFactory, LRSchedulerFactory, OptimizerFactory
from replay.models.nn.sequential.sasrec.dataset import SASRecPredictionBatch, SASRecTrainingBatch, SASRecValidationBatch
from replay.models.nn.sequential.sasrec.model import SASRecModel


# pylint: disable=too-many-instance-attributes
class SASRec(L.LightningModule):
    """
    SASRec Lightning module
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        tensor_schema: TensorSchema,
        block_count: int = 2,
        head_count: int = 1,
        embedding_dim: int = 50,
        max_seq_len: int = 200,
        dropout_rate: float = 0.2,
        ti_modification: bool = False,
        time_span: int = 256,
        loss_type: str = "CE",
        loss_sample_count: Optional[int] = None,
        negative_sampling_strategy: str = "global_uniform",
        negatives_sharing: bool = False,
        optimizer_factory: Optional[OptimizerFactory] = None,
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    ):
        """
        :param tensor_schema: Tensor schema of features.
        :param block_count: Number of Transformer blocks.
            Default: ``2``.
        :param head_count: Number of Attention heads.
            Default: ``1``.
        :param embedding_dim: Embedding dimension.
            Default: ``50``.
        :param max_seq_len: Max length of sequence.
            Default: ``200``.
        :param dropout_rate: Dropout rate.
            Default: ``0.2``.
        :param ti_modification: Enable time relation.
            Default: ``False``.
        :param time_span: Time span value.
            Default: ``256``.
        :param loss_type: Loss type. Possible values: ``"CE"``, ``"BCE"``.
            Default: ``CE``.
        :param loss_sample_count (Optional[int]): Sample count to calculate loss.
            Default: ``None``.
        :param negative_sampling_strategy: Negative sampling strategy to calculate loss on sampled negatives.
            Is used when large count of items in dataset.
            Possible values: ``"global_uniform"``, ``"inbatch"``
            Default: ``global_uniform``.
        :param negatives_sharing: Apply negative sharing in calculating sampled logits.
            Default: ``False``.
        :param optimizer_factory: Optimizer factory.
            Default: ``None``.
        :param lr_scheduler_factory: Learning rate schedule factory.
            Default: ``None``.
        """
        super().__init__()
        self.save_hyperparameters()
        self._model = SASRecModel(
            schema=tensor_schema,
            num_blocks=block_count,
            num_heads=head_count,
            embed_size=embedding_dim,
            max_len=max_seq_len,
            dropout=dropout_rate,
            ti_modification=ti_modification,
            time_span=time_span,
        )
        self._loss_type = loss_type
        self._loss_sample_count = loss_sample_count
        self._negative_sampling_strategy = negative_sampling_strategy
        self._negatives_sharing = negatives_sharing
        self._optimizer_factory = optimizer_factory
        self._lr_scheduler_factory = lr_scheduler_factory
        self._loss = self._create_loss()
        assert negative_sampling_strategy in {"global_uniform", "inbatch"}

        item_count = tensor_schema.item_id_features.item().cardinality
        assert item_count
        self._vocab_size = item_count

    # pylint: disable=unused-argument, arguments-differ
    def training_step(self, batch: SASRecTrainingBatch, batch_idx: int) -> torch.Tensor:
        """
        :param batch (SASRecTrainingBatch): Batch of training data.
        :param batch_idx (int): Batch index.

        :returns: Computed loss for batch.
        """
        if batch_idx % 100 == 0 and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # pylint: disable=arguments-differ
    def forward(self, feature_tensors: TensorMap, padding_mask: torch.BoolTensor) -> torch.Tensor:  # pragma: no cover
        """
        :param feature_tensors: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Calculated scores.
        """
        return self._model_predict(feature_tensors, padding_mask)

    # pylint: disable=unused-argument
    def predict_step(self, batch: SASRecPredictionBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        :param batch: Batch of prediction data.
        :param batch_idx: Batch index.
        :param dataloader_idx: Dataloader index.

        :returns: Calculated scores.
        """
        return self._model_predict(batch.features, batch.padding_mask)

    # pylint: disable=unused-argument, arguments-differ
    def validation_step(self, batch: SASRecValidationBatch, batch_idx: int) -> torch.Tensor:
        """
        :param batch (SASRecValidationBatch): Batch of prediction data.
        :param batch_idx (int): Batch index.

        :returns: Calculated scores.
        """
        return self._model_predict(batch.features, batch.padding_mask)

    def configure_optimizers(self) -> Any:
        """
        :returns: Configured optimizer and lr scheduler.
        """
        optimizer_factory = self._optimizer_factory or FatOptimizerFactory()
        optimizer = optimizer_factory.create(self._model.parameters())

        if self._lr_scheduler_factory is None:
            return optimizer

        lr_scheduler = self._lr_scheduler_factory.create(optimizer)
        return [optimizer], [lr_scheduler]

    def _model_predict(self, feature_tensors: TensorMap, padding_mask: torch.BoolTensor) -> torch.Tensor:
        model: SASRecModel
        if isinstance(self._model, torch.nn.DataParallel):
            model = cast(SASRecModel, self._model.module)  # multigpu
        else:
            model = self._model
        scores = model.predict(feature_tensors, padding_mask)
        return scores

    def _compute_loss(self, batch: SASRecTrainingBatch) -> torch.Tensor:
        if self._loss_type == "BCE":
            if self._loss_sample_count is None:
                loss_func = self._compute_loss_bce
            else:
                loss_func = self._compute_loss_bce_sampled
        elif self._loss_type == "CE":
            if self._loss_sample_count is None:
                loss_func = self._compute_loss_ce
            else:
                loss_func = self._compute_loss_ce_sampled
        else:
            raise ValueError(f"Not supported loss type: {self._loss_type}")

        loss = loss_func(
            batch.features,
            batch.labels,
            batch.padding_mask,
            batch.labels_padding_mask,
        )
        return loss

    def _compute_loss_bce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # [B x L x V]
        logits = self._model.forward(feature_tensors, padding_mask)

        # Take only logits which correspond to non-padded tokens
        # M = non_zero_count(target_padding_mask)
        logits = logits[target_padding_mask]  # [M x V]
        labels = positive_labels[target_padding_mask]  # [M]

        bce_labels = torch.zeros(
            (logits.size(0), logits.size(-1)),
            device=logits.device,
        )

        # Fill positives with ones, all negatives are zeros
        bce_labels.scatter_(
            dim=-1,
            index=labels.unsqueeze(-1),
            value=1,
        )

        loss = self._loss(logits, bce_labels) / logits.size(0)
        return loss

    def _compute_loss_bce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        (positive_logits, negative_logits, *_) = self._get_sampled_logits(
            feature_tensors, positive_labels, padding_mask, target_padding_mask
        )

        positive_prob = torch.sigmoid(positive_logits)
        negative_prob = torch.sigmoid(negative_logits)

        # Clamp and eps for numerical stability
        clamp_border: float = 100.0
        eps = 1e-6
        positive_loss = torch.clamp(torch.log((positive_prob) + eps), -clamp_border, clamp_border).sum()
        negative_loss = torch.clamp(torch.log((1 - negative_prob) + eps), -clamp_border, clamp_border).sum()

        loss = -(positive_loss + negative_loss)
        loss /= positive_logits.size(0)

        return loss

    def _compute_loss_ce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # logits: [B x L x V]
        logits = self._model.forward(
            feature_tensors,
            padding_mask,
        )

        # labels: [B x L]
        labels = positive_labels.masked_fill(mask=(~target_padding_mask), value=-100)

        logits_flat = logits.view(-1, logits.size(-1))  # [(B * L) x V]
        labels_flat = labels.view(-1)  # [(B * L)]

        loss = self._loss(logits_flat, labels_flat)
        return loss

    def _compute_loss_ce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        assert self._loss_sample_count is not None
        (positive_logits, negative_logits, positive_labels, negative_labels, vocab_size) = self._get_sampled_logits(
            feature_tensors, positive_labels, padding_mask, target_padding_mask
        )
        n_negative_samples = min(self._loss_sample_count, vocab_size)

        # Reject negative samples matching target label & correct for remaining samples
        reject_labels = positive_labels == negative_labels  # [masked_batch_seq_size x n_negative_samples]
        negative_logits += math.log(vocab_size - 1)
        negative_logits -= 1e6 * reject_labels
        negative_logits -= torch.log((n_negative_samples - reject_labels.sum(dim=-1, keepdim=True)).float())

        # Apply regular softmax cross entropy
        # [masked_batch_seq_size x (1 + n_negative_samples)]
        logits = torch.cat([positive_logits, negative_logits], dim=1).float()
        labels_flat = torch.zeros(positive_logits.size(0), dtype=torch.long, device=padding_mask.device)
        loss = self._loss(logits, labels_flat)
        return loss

    # pylint: disable=too-many-locals
    def _get_sampled_logits(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor, int]:
        assert self._loss_sample_count is not None
        n_negative_samples = self._loss_sample_count
        positive_labels = cast(
            torch.LongTensor, torch.masked_select(positive_labels, target_padding_mask)
        )  # (masked_batch_seq_size,)
        masked_batch_seq_size = positive_labels.size(0)
        device = padding_mask.device
        output_emb = self._model.forward_step(feature_tensors, padding_mask)[target_padding_mask]

        positive_labels = cast(torch.LongTensor, positive_labels.view(-1, 1))
        ids = torch.arange(masked_batch_seq_size, dtype=torch.long, device=device)
        unique_positive_labels, positive_labels_indices = positive_labels.unique(return_inverse=True)

        if self._negative_sampling_strategy == "global_uniform":
            vocab_size = self._vocab_size
            multinomial_sample_distribution = torch.ones(vocab_size, device=device)
            # positive_labels - 2d
            positive_logits = self._model.get_logits(output_emb, positive_labels)
        elif self._negative_sampling_strategy == "inbatch":
            positive_labels_indices = positive_labels_indices.view(masked_batch_seq_size, 1)
            # unique_positive_labels - 1d
            positive_logits = self._model.get_logits(output_emb, unique_positive_labels)
            vocab_size = unique_positive_labels.size(0)
            if self._negatives_sharing:
                multinomial_sample_distribution = torch.ones(vocab_size, device=device)
            else:
                multinomial_sample_distribution = torch.softmax(positive_logits, dim=-1)
        else:
            raise NotImplementedError(f"Unknown negative sampling strategy: {self._negative_sampling_strategy}")
        n_negative_samples = min(n_negative_samples, vocab_size)

        if self._negatives_sharing:
            negative_labels = torch.multinomial(
                multinomial_sample_distribution,
                num_samples=n_negative_samples,
                replacement=False,
            )
            negative_labels = negative_labels.unsqueeze(0).repeat(masked_batch_seq_size, 1)
        elif self._negative_sampling_strategy == "global_uniform":
            negative_labels = torch.randint(
                low=0,
                high=vocab_size,
                size=(masked_batch_seq_size, n_negative_samples),
                dtype=torch.long,
                device=device,
            )
        else:
            negative_labels = torch.multinomial(
                multinomial_sample_distribution,
                num_samples=n_negative_samples,
                replacement=False,
            )
        negative_labels = cast(torch.LongTensor, negative_labels)

        if self._negative_sampling_strategy == "global_uniform":
            if self._negatives_sharing:
                unique_negative_labels, negative_labels_indices = negative_labels.unique(return_inverse=True)
                negative_labels_indices = negative_labels_indices.view(masked_batch_seq_size, n_negative_samples)
                # unique_negative_labels - 1d
                negative_logits = self._model.get_logits(output_emb, unique_negative_labels)
                negative_logits = negative_logits[ids, negative_labels_indices.T].T
            else:
                # unique_negative_labels - 1d
                negative_logits = self._model.get_logits(output_emb, negative_labels)
        else:  # self._negative_sampling_strategy == "inbatch":
            negative_labels_indices = negative_labels
            negative_logits = positive_logits
            negative_logits = negative_logits[ids, negative_labels_indices.T].T
            positive_logits = positive_logits[ids, positive_labels_indices.T].T

        return (positive_logits, negative_logits, positive_labels, negative_labels, vocab_size)

    def _create_loss(self) -> Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]:
        if self._loss_type == "BCE":
            return torch.nn.BCEWithLogitsLoss(reduction="sum")

        if self._loss_type == "CE":
            return torch.nn.CrossEntropyLoss()

        raise NotImplementedError("Not supported loss_type")
