from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SCEParams:
    """Set of parameters for ScalableCrossEntropyLoss.

    Constructor arguments:
    :param n_buckets: Number of buckets into which samples will be distributed.
    :param bucket_size_x: Number of item hidden representations that will be in each bucket.
    :param bucket_size_y: Number of item embeddings that will be in each bucket.
    :param mix_x: Whether a randomly generated matrix will be multiplied by the model output matrix or not.
        Default: ``False``.
    """

    n_buckets: int
    bucket_size_x: int
    bucket_size_y: int
    mix_x: bool = False

    def _get_not_none_params(self):
        return [self.n_buckets, self.bucket_size_x, self.bucket_size_y]


class ScalableCrossEntropyLoss:
    def __init__(self, sce_params: SCEParams):
        """
        ScalableCrossEntropyLoss for Sequential Recommendations with Large Item Catalogs.
        Reference article may be found at https://arxiv.org/pdf/2409.18721.

        :param SCEParams: Dataclass with ScalableCrossEntropyLoss parameters.
            Dataclass contains following values:
                :param n_buckets: Number of buckets into which samples will be distributed.
                :param bucket_size_x: Number of item hidden representations that will be in each bucket.
                :param bucket_size_y: Number of item embeddings that will be in each bucket.
                :param mix_x: Whether a randomly generated matrix will be multiplied by the model output matrix or not.
                    Default: ``False``.
        """
        assert all(
            param is not None for param in sce_params._get_not_none_params()
        ), "You should define ``n_buckets``, ``bucket_size_x``, ``bucket_size_y`` when using SCE loss function."
        self._n_buckets = sce_params.n_buckets
        self._bucket_size_x = sce_params.bucket_size_x
        self._bucket_size_y = sce_params.bucket_size_y
        self._mix_x = sce_params.mix_x

    def __call__(
        self,
        embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        all_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
        tokens_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        ScalableCrossEntropyLoss computation.

        :param embeddings: Matrix of the last transformer block outputs.
        :param positive_labels: Positive labels.
        :param all_embeddings: Matrix of all item embeddings.
        :param padding_mask: Padding mask.
        :param tokens_mask: Tokens mask (need only for Bert4Rec).
            Default: ``None``.
        """
        masked_tokens = padding_mask if tokens_mask is None else ~(~padding_mask + tokens_mask)

        hd = torch.tensor(embeddings.shape[-1])
        x = embeddings.view(-1, hd)
        y = positive_labels.view(-1)
        w = all_embeddings

        correct_class_logits_ = (x * torch.index_select(w, dim=0, index=y)).sum(dim=1)  # (bs,)

        with torch.no_grad():
            if self._mix_x:
                omega = 1 / torch.sqrt(torch.sqrt(hd)) * torch.randn(x.shape[0], self._n_buckets, device=x.device)
                buckets = omega.T @ x
                del omega
            else:
                buckets = (
                    1 / torch.sqrt(torch.sqrt(hd)) * torch.randn(self._n_buckets, hd, device=x.device)
                )  # (n_b, hd)

        with torch.no_grad():
            x_bucket = buckets @ x.T  # (n_b, hd) x (hd, b) -> (n_b, b)
            x_bucket[:, ~padding_mask.view(-1)] = float("-inf")
            _, top_x_bucket = torch.topk(x_bucket, dim=1, k=self._bucket_size_x)  # (n_b, bs_x)
            del x_bucket

            y_bucket = buckets @ w.T  # (n_b, hd) x (hd, n_cl) -> (n_b, n_cl)

            _, top_y_bucket = torch.topk(y_bucket, dim=1, k=self._bucket_size_y)  # (n_b, bs_y)
            del y_bucket

        x_bucket = torch.gather(x, 0, top_x_bucket.view(-1, 1).expand(-1, hd)).view(
            self._n_buckets, self._bucket_size_x, hd
        )  # (n_b, bs_x, hd)
        y_bucket = torch.gather(w, 0, top_y_bucket.view(-1, 1).expand(-1, hd)).view(
            self._n_buckets, self._bucket_size_y, hd
        )  # (n_b, bs_y, hd)

        wrong_class_logits = x_bucket @ y_bucket.transpose(-1, -2)  # (n_b, bs_x, bs_y)
        mask = (
            torch.index_select(y, dim=0, index=top_x_bucket.view(-1)).view(self._n_buckets, self._bucket_size_x)[
                :, :, None
            ]
            == top_y_bucket[:, None, :]
        )  # (n_b, bs_x, bs_y)
        wrong_class_logits = wrong_class_logits.masked_fill(mask, float("-inf"))  # (n_b, bs_x, bs_y)
        correct_class_logits = torch.index_select(correct_class_logits_, dim=0, index=top_x_bucket.view(-1)).view(
            self._n_buckets, self._bucket_size_x
        )[
            :, :, None
        ]  # (n_b, bs_x, 1)
        logits = torch.cat((wrong_class_logits, correct_class_logits), dim=2)  # (n_b, bs_x, bs_y + 1)

        loss_ = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            (logits.shape[-1] - 1)
            * torch.ones(logits.shape[0] * logits.shape[1], dtype=torch.int64, device=logits.device),
            reduction="none",
        )  # (n_b * bs_x,)
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        loss.scatter_reduce_(0, top_x_bucket.view(-1), loss_, reduce="amax", include_self=False)
        loss = loss[(loss != 0) & (masked_tokens).view(-1)]
        loss = torch.mean(loss)

        return loss
