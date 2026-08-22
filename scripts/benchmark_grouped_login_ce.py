"""Benchmark packed GroupedLogInCESampled training against gradient accumulation."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from replay.nn.loss import GroupedLogInCESampled, LogInCESampled


@dataclass(frozen=True)
class BenchmarkConfig:
    logical_batches: int = 5
    logical_batch_size: int = 128
    sequence_length: int = 50
    embedding_dim: int = 192
    attention_heads: int = 16
    encoder_layers: int = 8
    positives: int = 2
    negatives: int = 20_000
    cardinality: int = 100_000
    warmup_steps: int = 3
    benchmark_steps: int = 10
    seed: int = 777


@dataclass(frozen=True)
class Batch:
    tokens: torch.LongTensor
    positives: torch.LongTensor
    negatives: torch.LongTensor
    padding_mask: torch.BoolTensor
    target_padding_mask: torch.BoolTensor


class _TrainingModel(torch.nn.Module):
    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.cardinality, config.embedding_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.embedding_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.projection = torch.nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.item_embeddings = torch.nn.Embedding(config.cardinality, config.embedding_dim)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        return self.projection(self.encoder(self.token_embeddings(tokens)))

    def score(self, query: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        candidates = self.item_embeddings(labels)
        if labels.dim() == 1:
            return query @ candidates.T
        return torch.einsum("bd,bnd->bn", query, candidates)


def _make_batch(config: BenchmarkConfig, device: torch.device) -> Batch:
    generator = torch.Generator(device=device).manual_seed(config.seed)
    physical_batch_size = config.logical_batches * config.logical_batch_size
    return Batch(
        tokens=torch.randint(
            0,
            config.cardinality,
            (physical_batch_size, config.sequence_length),
            generator=generator,
            device=device,
        ),
        positives=torch.randint(
            0,
            config.cardinality,
            (physical_batch_size, config.sequence_length, config.positives),
            generator=generator,
            device=device,
        ),
        negatives=torch.randint(
            0,
            config.cardinality,
            (config.logical_batches, config.negatives),
            generator=generator,
            device=device,
        ),
        padding_mask=torch.ones(
            physical_batch_size,
            config.sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        target_padding_mask=torch.ones(
            physical_batch_size,
            config.sequence_length,
            config.positives,
            dtype=torch.bool,
            device=device,
        ),
    )


def _reference_step(
    model: _TrainingModel,
    loss: LogInCESampled,
    batch: Batch,
    config: BenchmarkConfig,
    *,
    use_bfloat16: bool,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=batch.tokens.device)
    for group_index in range(config.logical_batches):
        start = group_index * config.logical_batch_size
        end = start + config.logical_batch_size
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            embeddings = model(batch.tokens[start:end])
            group_loss = loss(
                embeddings,
                {},
                batch.positives[start:end],
                batch.negatives[group_index],
                batch.padding_mask[start:end],
                batch.target_padding_mask[start:end],
            )
            scaled_loss = group_loss / config.logical_batches
        scaled_loss.backward()
        total_loss = total_loss + scaled_loss.detach()
    return total_loss


def _packed_step(
    model: _TrainingModel,
    loss: GroupedLogInCESampled,
    batch: Batch,
    *,
    use_bfloat16: bool,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        embeddings = model(batch.tokens)
        packed_loss = loss(
            embeddings,
            {},
            batch.positives,
            batch.negatives,
            batch.padding_mask,
            batch.target_padding_mask,
        )
    packed_loss.backward()
    return packed_loss.detach()


def _parity_probe(device: torch.device, seed: int) -> dict[str, float]:
    config = BenchmarkConfig(
        logical_batches=3,
        logical_batch_size=4,
        sequence_length=8,
        embedding_dim=48,
        attention_heads=4,
        encoder_layers=1,
        positives=2,
        negatives=64,
        cardinality=1_024,
        warmup_steps=0,
        benchmark_steps=1,
        seed=seed,
    )
    torch.manual_seed(seed)
    reference_model = _TrainingModel(config).to(device)
    packed_model = copy.deepcopy(reference_model)
    batch = _make_batch(config, device)
    reference_loss = LogInCESampled()
    reference_loss.logits_callback = reference_model.score
    packed_loss = GroupedLogInCESampled(config.logical_batch_size)
    packed_loss.logits_callback = packed_model.score

    torch.backends.cuda.matmul.allow_tf32 = False
    expected = _reference_step(
        reference_model,
        reference_loss,
        batch,
        config,
        use_bfloat16=False,
    )
    actual = _packed_step(
        packed_model,
        packed_loss,
        batch,
        use_bfloat16=False,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    max_gradient_difference = 0.0
    for expected_parameter, actual_parameter in zip(
        reference_model.parameters(),
        packed_model.parameters(),
        strict=True,
    ):
        if expected_parameter.grad is None or actual_parameter.grad is None:
            msg = "Parity probe produced a missing gradient."
            raise RuntimeError(msg)
        max_gradient_difference = max(
            max_gradient_difference,
            (expected_parameter.grad - actual_parameter.grad).abs().max().item(),
        )
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=2e-4,
            atol=2e-6,
        )
    return {
        "reference_loss": expected.item(),
        "packed_loss": actual.item(),
        "loss_absolute_difference": abs(expected.item() - actual.item()),
        "max_gradient_absolute_difference": max_gradient_difference,
    }


def _measure(step, warmup_steps: int, benchmark_steps: int) -> list[float]:
    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()

    elapsed_ms = []
    for _ in range(benchmark_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
    return elapsed_ms


def _peak_allocated_mib(model: _TrainingModel, step) -> float:
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    model.zero_grad(set_to_none=True)
    return peak


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def run(config: BenchmarkConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        msg = "This benchmark requires a CUDA device."
        raise RuntimeError(msg)
    if not torch.cuda.is_bf16_supported():
        msg = "This benchmark requires CUDA bfloat16 support."
        raise RuntimeError(msg)

    device = torch.device("cuda")
    parity = _parity_probe(device, config.seed)
    torch.manual_seed(config.seed)
    model = _TrainingModel(config).to(device)
    batch = _make_batch(config, device)
    reference_loss = LogInCESampled()
    reference_loss.logits_callback = model.score
    packed_loss = GroupedLogInCESampled(config.logical_batch_size)
    packed_loss.logits_callback = model.score

    def reference_step() -> torch.Tensor:
        return _reference_step(
            model,
            reference_loss,
            batch,
            config,
            use_bfloat16=True,
        )

    def packed_step() -> torch.Tensor:
        return _packed_step(
            model,
            packed_loss,
            batch,
            use_bfloat16=True,
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    reference_ms = _measure(reference_step, config.warmup_steps, config.benchmark_steps)
    packed_ms = _measure(packed_step, config.warmup_steps, config.benchmark_steps)
    reference_peak_mib = _peak_allocated_mib(model, reference_step)
    packed_peak_mib = _peak_allocated_mib(model, packed_step)
    reference_median = statistics.median(reference_ms)
    packed_median = statistics.median(packed_ms)
    users = config.logical_batches * config.logical_batch_size
    return {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "parity": parity,
        "reference": {
            "median_step_ms": reference_median,
            "p25_step_ms": _percentile(reference_ms, 0.25),
            "p75_step_ms": _percentile(reference_ms, 0.75),
            "users_per_second": users * 1000 / reference_median,
            "peak_allocated_mib": reference_peak_mib,
        },
        "packed": {
            "median_step_ms": packed_median,
            "p25_step_ms": _percentile(packed_ms, 0.25),
            "p75_step_ms": _percentile(packed_ms, 0.75),
            "users_per_second": users * 1000 / packed_median,
            "peak_allocated_mib": packed_peak_mib,
        },
        "speedup": reference_median / packed_median,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.warmup_steps < 0 or args.benchmark_steps <= 0:
        msg = "warmup-steps must be non-negative and benchmark-steps must be positive."
        raise ValueError(msg)
    config = BenchmarkConfig(
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
    )
    payload = json.dumps(run(config), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
