import gc
import math
from typing import Any, Optional, Tuple, cast

import torch
from torch import nn

from replay.data.nn import TensorMap, TensorSchema
from replay.models.nn.optimizer_utils import FatOptimizerFactory, LRSchedulerFactory, OptimizerFactory
from .dataset import SasRecLLMTrainingBatch

from .model import SasRecLLMModel
from ..sasrec.lightning import SasRec


class SasRecLLM(SasRec):
    """
    SASRecLLM Lightning module.

    You can get initialization parameters with attribute `hparams`
    for object of SasRecLLM instance.
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        profile_emb_dim: int,
        block_count: int = 2,
        head_count: int = 1,
        hidden_size: int = 50,
        max_seq_len: int = 200,
        dropout_rate: float = 0.2,
        ti_modification: bool = False,
        time_span: int = 256,
        loss_type: str = "CE",
        loss_sample_count: Optional[int] = None,
        negative_sampling_strategy: str = "global_uniform",
        negatives_sharing: bool = False,
        optimizer_factory: OptimizerFactory = FatOptimizerFactory(),
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
        scale_guide_loss: bool = False,
        alpha: float = 0.8,
        profile_distil_epochs: int = 0,
        reconstruction_layer: int = -1,
        criterion_reconstruct_name: str = 'MSE',
        weighting_scheme='mean',
        use_down_scale=True,
        use_upscale=False,
        weight_scale=None,
        multi_profile=False,
        multi_profile_aggr_scheme='mean'
    ) -> None:
        """
        :param tensor_schema: Tensor schema of features.
        :param block_count: Number of Transformer blocks.
            Default: ``2``.
        :param head_count: Number of Attention heads.
            Default: ``1``.
        :param hidden_size: Hidden size of transformer.
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
            Default: ``FatOptimizerFactory``.
        :param lr_scheduler_factory: Learning rate schedule factory.
            Default: ``None``.
        :param scale_guide_loss: Scale guide loss.
            Default: ``False``.
        :param alpha: Alpha value for loss.
            Default: ``0.8``.
        :param profile_distil_epochs: Profile distillation epochs.
            Default: ``0``.
        :param reconstruction_layer: Layer for reconstruction.
            Default: ``-1``.
        :param criterion_reconstruct_name: Name of criterion for reconstruction.
            Default: ``MSE``.
        :param weighting_scheme: Weighting scheme.
            Default: ``mean``.
        :param use_down_scale: Use down scale.
            Default: ``True``.
        :param use_upscale: Use upscale.
            Default: ``False``.
        :param weight_scale: Weight scale.
            Default: ``None``.
        :param multi_profile: Multi profile.
            Default: ``False``.
        :param multi_profile_aggr_scheme: Multi profile aggregation scheme.
            Default: ``mean``.
        """
        super().__init__(
            tensor_schema=tensor_schema,
            block_count=block_count,
            head_count=head_count,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
            ti_modification=ti_modification,
            time_span=time_span,
            loss_type=loss_type,
            loss_sample_count=loss_sample_count,
            negative_sampling_strategy=negative_sampling_strategy,
            negatives_sharing=negatives_sharing,
            optimizer_factory=optimizer_factory,
            lr_scheduler_factory=lr_scheduler_factory,
        )
        # override SASRec model
        del self._model
        gc.collect()
        self._model = SasRecLLMModel(schema=tensor_schema,
                                     profile_emb_dim=profile_emb_dim,
                                     num_blocks=block_count,
                                     num_heads=head_count,
                                     hidden_size=hidden_size,
                                     max_len=max_seq_len,
                                     dropout=dropout_rate,
                                     ti_modification=ti_modification,
                                     reconstruction_layer=reconstruction_layer,
                                     weighting_scheme=weighting_scheme,
                                     use_down_scale=use_down_scale,
                                     use_upscale=use_upscale,
                                     weight_scale=weight_scale,
                                     multi_profile=multi_profile,
                                     multi_profile_aggr_scheme=multi_profile_aggr_scheme)
        self.scale_guide_loss = scale_guide_loss
        self.alpha = alpha
        self.profile_distil_epochs = profile_distil_epochs
        self.criterion_reconstruct = self._init_criterion_reconstruct(criterion_reconstruct_name)

    def _init_criterion_reconstruct(self, criterion_name: str) -> Any:
        """
        :param criterion_name: Name of criterion for reconstruction.
        :return: Criterion for reconstruction
        """
        if criterion_name == 'MSE':
            return lambda x,y: nn.MSELoss()(x,y)
        if criterion_name == 'RMSE':
            return lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
        if criterion_name == 'CosSim':
            return lambda x,y: 1 - torch.mean(nn.CosineSimilarity(dim=1, eps=1e-6)(x,y))
        raise Exception('Not existing reconstruction loss')

    def _compute_loss(self, batch: SasRecLLMTrainingBatch) -> torch.Tensor:
        if self._loss_type == "BCE":
            loss_func = self._compute_loss_bce if self._loss_sample_count is None else self._compute_loss_bce_sampled
        elif self._loss_type == "CE":
            loss_func = self._compute_loss_ce if self._loss_sample_count is None else self._compute_loss_ce_sampled
        else:
            msg = f"Not supported loss type: {self._loss_type}"
            raise ValueError(msg)

        # returns hidden states additionally
        loss_model, hidden_state = loss_func(
            batch.features,
            batch.labels,
            batch.padding_mask,
            batch.labels_padding_mask,
        )

        epoch = self.trainer.current_epoch
        if epoch >= self.profile_distil_epochs:
            return loss_model
        # reconstruction loss
        loss_guide = self._compute_loss_reconstruction(
            batch.user_profile_embeddings_batch,
            hidden_state,
            batch.existing_profile_binary_mask_batch,
        )
        return self._merge_losses(loss_model, loss_guide)

    def _merge_losses(self, loss_model: torch.Tensor, loss_guide: torch.Tensor) -> torch.Tensor:
        """
        :param loss_model: Model loss
        :param loss_guide: Guide loss

        :return: Merged loss
        """
        if self.scale_guide_loss:
            loss_model_value = loss_model.item()
            loss_guide_value = loss_guide.item()
            eps = 1e-8

            scale_for_guide = loss_model_value / (loss_guide_value + eps)
            loss_guide = loss_guide * scale_for_guide
        return self.alpha * loss_guide + (1 - self.alpha) * loss_model

    def _compute_loss_reconstruction(
        self,
        user_profile_emb: torch.Tensor,
        hidden_for_reconstruction: torch.Tensor,
        existing_profile_binary_mask_batch: torch.BoolTensor,
    ) -> torch.Tensor:
        user_profile_emb_transformed = self._model.aggregate_profile(user_profile_emb)
        if self._model.use_upscale:
            hidden_for_reconstruction = self._model.hidden_layer_transform(hidden_for_reconstruction)

        existing_profile_binary_mask_batch = existing_profile_binary_mask_batch.flatten()

        loss_guide = self.criterion_reconstruct(hidden_for_reconstruction[existing_profile_binary_mask_batch],
                                              user_profile_emb_transformed[existing_profile_binary_mask_batch])
        return loss_guide

    def _compute_loss_bce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B x L x V]
        logits, hidden_state = self._model.forward(feature_tensors, padding_mask)

        """
        Take only logits which correspond to non-padded tokens
        M = non_zero_count(target_padding_mask)
        """
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
        return loss, hidden_state

    def _compute_loss_bce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (positive_logits, negative_logits, hidden_state, *_) = self._get_sampled_logits(
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

        return loss, hidden_state

    def _compute_loss_ce(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits: [B x L x V]
        logits, hidden_state = self._model.forward(
            feature_tensors,
            padding_mask,
        )

        # labels: [B x L]
        labels = positive_labels.masked_fill(mask=(~target_padding_mask), value=-100)

        logits_flat = logits.view(-1, logits.size(-1))  # [(B * L) x V]
        labels_flat = labels.view(-1)  # [(B * L)]

        loss = self._loss(logits_flat, labels_flat)
        return loss, hidden_state

    def _compute_loss_ce_sampled(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._loss_sample_count is not None
        (positive_logits, negative_logits, hidden_state, positive_labels, negative_labels, vocab_size) = self._get_sampled_logits(
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
        return loss, hidden_state

    def _get_sampled_logits(
        self,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor, int]:
        assert self._loss_sample_count is not None
        n_negative_samples = self._loss_sample_count
        positive_labels = cast(
            torch.LongTensor, torch.masked_select(positive_labels, target_padding_mask)
        )  # (masked_batch_seq_size,)
        masked_batch_seq_size = positive_labels.size(0)
        device = padding_mask.device
        output_emb, hidden_state = self._model.forward_step(feature_tensors, padding_mask)
        output_emb = output_emb[target_padding_mask]

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
            msg = f"Unknown negative sampling strategy: {self._negative_sampling_strategy}"
            raise NotImplementedError(msg)
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

        return (positive_logits, negative_logits, hidden_state, positive_labels, negative_labels, vocab_size)
