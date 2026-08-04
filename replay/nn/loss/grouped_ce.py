from collections.abc import Callable

import torch
import torch.nn.functional as functional

from replay.data.nn import TensorMap
from replay.nn.head import EmbeddingTyingHead
from replay.nn.loss.base import mask_negative_logits
from replay.nn.loss.ce import CESampled
from replay.nn.sequential.twotower import TwoTower


class GroupedCESampled(CESampled):
    """Sampled CE with one independent negative pool per logical batch group.

    Negative pools must contain unique item IDs. Group losses are divided by
    ``groups_per_batch`` even for the final short
    batch. This preserves the weighting of the original smaller batches when
    the physical batch size is increased together with gradient accumulation.
    """

    def __init__(
        self,
        logical_batch_size: int,
        groups_per_batch: int,
        expected_num_negatives: int | None = None,
        cardinality: int | None = None,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ) -> None:
        """
        :param logical_batch_size: Number of rows that share one negative pool.
        :param groups_per_batch: Number of logical groups in a full physical batch.
        :param expected_num_negatives: Optional required size of every negative pool.
        :param cardinality: Optional catalog size enabling collision masking without a
            target-by-negative comparison matrix.
        :param negative_labels_ignore_index: Value ignored in negative labels.
        :param kwargs: Arguments passed to :class:`torch.nn.CrossEntropyLoss`.
        """
        if logical_batch_size <= 0:
            msg = "The logical_batch_size parameter must be positive."
            raise ValueError(msg)
        if groups_per_batch <= 1:
            msg = "The groups_per_batch parameter must be greater than one."
            raise ValueError(msg)
        if expected_num_negatives is not None and expected_num_negatives <= 0:
            msg = "The expected_num_negatives parameter must be positive."
            raise ValueError(msg)
        if cardinality is not None and cardinality <= 0:
            msg = "The cardinality parameter must be positive."
            raise ValueError(msg)
        super().__init__(negative_labels_ignore_index=negative_labels_ignore_index, **kwargs)
        if self._loss.reduction != "mean":
            msg = "GroupedCESampled supports only mean reduction."
            raise ValueError(msg)
        self.logical_batch_size = logical_batch_size
        self.groups_per_batch = groups_per_batch
        self.expected_num_negatives = expected_num_negatives
        self.cardinality = cardinality
        self.register_buffer("_negative_column_lookup", None, persistent=False)

    def _validate_inputs(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> int:
        if model_embeddings.dim() != 3:
            msg = "model_embeddings must have shape [batch, sequence, embedding]."
            raise ValueError(msg)
        if positive_labels.dim() != 3 or positive_labels.size(-1) != 1:
            msg = "GroupedCESampled requires exactly one positive label per sequence position."
            raise ValueError(msg)
        if positive_labels.shape[:2] != model_embeddings.shape[:2]:
            msg = "positive_labels and model_embeddings must have equal batch and sequence dimensions."
            raise ValueError(msg)
        if target_padding_mask.shape != positive_labels.shape:
            msg = "target_padding_mask must have the same shape as positive_labels."
            raise ValueError(msg)
        if negative_labels.dim() != 2 or negative_labels.size(0) != self.groups_per_batch:
            msg = f"negative_labels must have shape [{self.groups_per_batch}, num_negatives]."
            raise ValueError(msg)
        if self.expected_num_negatives is not None and negative_labels.size(1) != self.expected_num_negatives:
            msg = f"Each negative pool must contain {self.expected_num_negatives} items, got {negative_labels.size(1)}."
            raise ValueError(msg)
        if model_embeddings.size(0) == 0:
            msg = "GroupedCESampled does not support empty batches."
            raise ValueError(msg)

        active_groups = (model_embeddings.size(0) + self.logical_batch_size - 1) // self.logical_batch_size
        if active_groups > self.groups_per_batch:
            capacity = self.logical_batch_size * self.groups_per_batch
            msg = f"Batch size {model_embeddings.size(0)} exceeds grouped loss capacity {capacity}."
            raise ValueError(msg)
        return active_groups

    def _loss_from_logits(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
    ) -> torch.Tensor:
        negative_logits = self._mask_negative_logits(negative_logits, negative_labels, positive_labels)
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        target = torch.zeros(positive_logits.size(0), dtype=torch.long, device=logits.device)
        return self._loss(logits, target)

    def _mask_negative_logits(
        self,
        negative_logits: torch.Tensor,
        negative_labels: torch.LongTensor,
        positive_labels: torch.LongTensor,
    ) -> torch.Tensor:
        if self.cardinality is None:
            return mask_negative_logits(
                negative_logits,
                negative_labels,
                positive_labels.unsqueeze(-1),
                self.negative_labels_ignore_index,
            )

        lookup = self._negative_column_lookup
        if lookup is None or lookup.device != negative_labels.device:
            lookup = torch.empty(self.cardinality, dtype=torch.int32, device=negative_labels.device)
            self._negative_column_lookup = lookup
        lookup.fill_(-1)

        valid_negatives = negative_labels.ge(0) & negative_labels.lt(self.cardinality)
        negative_columns = torch.arange(negative_labels.numel(), dtype=lookup.dtype, device=negative_labels.device)
        lookup[negative_labels[valid_negatives]] = negative_columns[valid_negatives]

        valid_positives = positive_labels.ge(0) & positive_labels.lt(self.cardinality)
        safe_positives = positive_labels.clamp(min=0, max=self.cardinality - 1)
        collision_columns = lookup[safe_positives].long()
        collision_rows = torch.arange(positive_labels.numel(), device=positive_labels.device)
        collisions = valid_positives & collision_columns.ge(0)
        masked_value = max(-1e9, torch.finfo(negative_logits.dtype).min)
        negative_logits.masked_fill_(
            negative_labels.eq(self.negative_labels_ignore_index).unsqueeze(0),
            masked_value,
        )
        negative_logits[collision_rows[collisions], collision_columns[collisions]] = masked_value
        return negative_logits

    def _score_group(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
    ) -> torch.Tensor:
        positive_logits = self.logits_callback(model_embeddings, positive_labels.unsqueeze(-1))
        scoring_negative_labels = negative_labels.masked_fill(
            negative_labels.eq(self.negative_labels_ignore_index),
            0,
        )
        negative_logits = self.logits_callback(model_embeddings, scoring_negative_labels)
        return self._loss_from_logits(
            positive_logits,
            negative_logits,
            positive_labels,
            negative_labels,
        )

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        active_groups = self._validate_inputs(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_padding_mask,
        )
        loss = model_embeddings.new_zeros(())
        for group_index in range(active_groups):
            start = group_index * self.logical_batch_size
            end = min(start + self.logical_batch_size, model_embeddings.size(0))
            active_targets = target_padding_mask[start:end].squeeze(-1)
            group_embeddings = model_embeddings[start:end][active_targets]
            group_positive_labels = positive_labels[start:end].squeeze(-1)[active_targets]
            if group_embeddings.size(0) == 0:
                msg = "Each active logical group must contain at least one target."
                raise ValueError(msg)
            loss = loss + self._score_group(
                group_embeddings,
                group_positive_labels,
                negative_labels[group_index],
            )
        return loss / self.groups_per_batch


class _CatalogGradientState:
    def __init__(self, accumulation_steps: int) -> None:
        if accumulation_steps <= 1:
            msg = "The accumulation_steps parameter must be greater than one."
            raise ValueError(msg)
        self.accumulation_steps = accumulation_steps
        self.catalog_embeddings: torch.Tensor | None = None
        self.pending_gradient: torch.Tensor | None = None
        self.prepared_window_end: bool | None = None
        self.retain_graph: bool | None = None
        self.window_global_step: int | None = None
        self.microbatch_count = 0

    def prepare(
        self,
        *,
        is_window_end: bool,
        trainer_accumulation_steps: int,
        global_step: int,
    ) -> None:
        if trainer_accumulation_steps != self.accumulation_steps:
            msg = (
                f"The loss expects {self.accumulation_steps} accumulation steps, "
                f"but the trainer uses {trainer_accumulation_steps}."
            )
            raise ValueError(msg)
        if self.prepared_window_end is not None:
            msg = "The previously prepared catalog-cache microbatch was not consumed."
            raise RuntimeError(msg)
        if self.window_global_step is not None and self.window_global_step != global_step:
            msg = "global_step changed inside a catalog-cache accumulation window."
            raise RuntimeError(msg)
        next_microbatch_count = self.microbatch_count + 1
        if next_microbatch_count > self.accumulation_steps:
            msg = "The catalog-cache accumulation window did not finish on time."
            raise RuntimeError(msg)
        if next_microbatch_count == self.accumulation_steps and not is_window_end:
            msg = "The catalog-cache window must end after accumulation_steps microbatches."
            raise RuntimeError(msg)
        if self.window_global_step is None:
            self.window_global_step = global_step
        self.microbatch_count = next_microbatch_count
        self.prepared_window_end = is_window_end

    def consume(self) -> bool:
        if self.prepared_window_end is None:
            msg = "Call prepare_accumulation_microbatch before the catalog-cache loss."
            raise RuntimeError(msg)
        is_window_end = self.prepared_window_end
        self.prepared_window_end = None
        self.retain_graph = not is_window_end
        return is_window_end

    def accumulate_gradient(
        self,
        gradient: torch.Tensor,
        *,
        is_window_end: bool,
    ) -> torch.Tensor | None:
        gradient = gradient.detach()
        self.retain_graph = None
        if self.pending_gradient is None:
            accumulated = gradient.clone()
        else:
            if self.pending_gradient.shape != gradient.shape:
                msg = "Catalog gradients changed shape inside an accumulation window."
                raise RuntimeError(msg)
            self.pending_gradient.add_(gradient)
            accumulated = self.pending_gradient
        if not is_window_end:
            self.pending_gradient = accumulated
            return None

        self.catalog_embeddings = None
        self.pending_gradient = None
        self.window_global_step = None
        self.microbatch_count = 0
        return accumulated

    def should_retain_graph(self) -> bool:
        if self.retain_graph is None:
            msg = "The catalog-cache forward pass did not prepare backward."
            raise RuntimeError(msg)
        return self.retain_graph

    def assert_backward_completed(self) -> None:
        if self.prepared_window_end is not None or self.retain_graph is not None:
            msg = "The catalog-cache forward pass did not complete exactly one backward pass."
            raise RuntimeError(msg)

    def assert_idle(self) -> None:
        if (
            any(
                value is not None
                for value in (
                    self.catalog_embeddings,
                    self.pending_gradient,
                    self.prepared_window_end,
                    self.retain_graph,
                    self.window_global_step,
                )
            )
            or self.microbatch_count != 0
        ):
            msg = "The catalog-cache accumulation state is not idle."
            raise RuntimeError(msg)


class _DeferredCatalogGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        catalog_embeddings: torch.Tensor,
        state: _CatalogGradientState,
        is_window_end: bool,
    ) -> torch.Tensor:
        ctx.state = state
        ctx.is_window_end = is_window_end
        return catalog_embeddings.view_as(catalog_embeddings)

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        return (
            ctx.state.accumulate_gradient(gradient, is_window_end=ctx.is_window_end),
            None,
            None,
        )


class CatalogCachedGroupedCESampled(GroupedCESampled):
    """Reuse one item-catalog graph across a gradient-accumulation window.

    The item tower must be deterministic and candidate-separable. The cache is
    differentiable: gradients from all microbatches reach the item tower once,
    at the optimizer-step boundary.
    """

    def __init__(
        self,
        cardinality: int,
        logical_batch_size: int,
        groups_per_batch: int,
        accumulation_steps: int,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ) -> None:
        """
        :param cardinality: Number of items returned by the item tower for the full catalog.
        :param logical_batch_size: Number of rows that share one negative pool.
        :param groups_per_batch: Number of logical groups in a full physical batch.
        :param accumulation_steps: Number of physical batches in one optimizer step.
        :param negative_labels_ignore_index: Value ignored in negative labels.
        :param kwargs: Arguments passed to :class:`torch.nn.CrossEntropyLoss`.
        """
        if cardinality <= 0:
            msg = "The cardinality parameter must be positive."
            raise ValueError(msg)
        super().__init__(
            logical_batch_size=logical_batch_size,
            groups_per_batch=groups_per_batch,
            cardinality=cardinality,
            negative_labels_ignore_index=negative_labels_ignore_index,
            **kwargs,
        )
        self.cardinality = cardinality
        self._catalog_state = _CatalogGradientState(accumulation_steps)

    def prepare_accumulation_microbatch(
        self,
        *,
        is_window_end: bool,
        trainer_accumulation_steps: int,
        global_step: int,
    ) -> None:
        self._catalog_state.prepare(
            is_window_end=is_window_end,
            trainer_accumulation_steps=trainer_accumulation_steps,
            global_step=global_step,
        )

    def should_retain_graph_for_current_backward(self) -> bool:
        return self._catalog_state.should_retain_graph()

    def assert_current_backward_completed(self) -> None:
        self._catalog_state.assert_backward_completed()

    def assert_accumulation_cache_idle(self) -> None:
        self._catalog_state.assert_idle()

    def _resolve_model_parts(self) -> tuple[torch.nn.Module, EmbeddingTyingHead]:
        callback: Callable = self.logits_callback
        owner = getattr(callback, "__self__", None)
        function = getattr(callback, "__func__", None)
        item_tower = getattr(getattr(owner, "body", None), "item_tower", None)
        head = getattr(owner, "head", None)
        if function is not TwoTower.get_logits or not isinstance(item_tower, torch.nn.Module):
            msg = "CatalogCachedGroupedCESampled requires a bound TwoTower.get_logits callback."
            raise TypeError(msg)
        if type(head) is not EmbeddingTyingHead:
            msg = "CatalogCachedGroupedCESampled supports the standard EmbeddingTyingHead only."
            raise TypeError(msg)
        return item_tower, head

    @staticmethod
    def _validate_item_tower(item_tower: torch.nn.Module) -> None:
        stochastic_modules = (
            torch.nn.AlphaDropout,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.Dropout,
            torch.nn.Dropout1d,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
            torch.nn.FeatureAlphaDropout,
            torch.nn.SyncBatchNorm,
        )
        unsupported = tuple(
            module.__class__.__name__ for module in item_tower.modules() if isinstance(module, stochastic_modules)
        )
        if unsupported:
            msg = f"The cached item tower must be deterministic and candidate-separable; found {unsupported}."
            raise TypeError(msg)

    def _get_catalog(self, item_tower: torch.nn.Module, device: torch.device) -> torch.Tensor:
        catalog = self._catalog_state.catalog_embeddings
        if catalog is None:
            self._validate_item_tower(item_tower)
            catalog = item_tower()
            if catalog.dim() != 2 or catalog.size(0) != self.cardinality:
                msg = f"The item tower must return a two-dimensional full catalog with {self.cardinality} rows."
                raise RuntimeError(msg)
            if catalog.device != device:
                msg = "The item catalog and model embeddings must be on the same device."
                raise RuntimeError(msg)
            self._catalog_state.catalog_embeddings = catalog
        elif catalog.device != device:
            msg = "The catalog cache changed device inside an accumulation window."
            raise RuntimeError(msg)
        return catalog

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        active_groups = self._validate_inputs(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_padding_mask,
        )
        is_window_end = self._catalog_state.consume()
        item_tower, _ = self._resolve_model_parts()
        catalog = self._get_catalog(item_tower, model_embeddings.device)
        catalog = _DeferredCatalogGradient.apply(catalog, self._catalog_state, is_window_end)

        loss = model_embeddings.new_zeros(())
        for group_index in range(active_groups):
            start = group_index * self.logical_batch_size
            end = min(start + self.logical_batch_size, model_embeddings.size(0))
            active_targets = target_padding_mask[start:end].squeeze(-1)
            group_embeddings = model_embeddings[start:end][active_targets]
            group_positive_labels = positive_labels[start:end].squeeze(-1)[active_targets]
            if group_embeddings.size(0) == 0:
                msg = "Each active logical group must contain at least one target."
                raise ValueError(msg)

            group_negative_labels = negative_labels[group_index]
            scoring_negative_labels = group_negative_labels.masked_fill(
                group_negative_labels.eq(self.negative_labels_ignore_index),
                0,
            )
            positive_logits = (group_embeddings * catalog[group_positive_labels]).sum(
                dim=-1,
                keepdim=True,
            )
            negative_logits = functional.linear(group_embeddings, catalog[scoring_negative_labels])
            loss = loss + self._loss_from_logits(
                positive_logits,
                negative_logits,
                group_positive_labels,
                group_negative_labels,
            )
        return loss / self.groups_per_batch
