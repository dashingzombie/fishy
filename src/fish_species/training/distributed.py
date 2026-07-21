"""Small single-node DistributedDataParallel runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialise_distributed(config: dict) -> DistributedContext:
    runtime = (config.get("training", {}).get("distributed", {}) or {})
    requested = bool(runtime.get("enabled", False))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = requested or env_world_size > 1

    if enabled and env_world_size <= 1:
        raise RuntimeError(
            "training.distributed.enabled=true requires torchrun (WORLD_SIZE > 1)"
        )

    if enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        if not torch.cuda.is_available():
            backend = str(runtime.get("backend", "gloo"))
            device = torch.device("cpu")
        else:
            backend = str(runtime.get("backend", "nccl"))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(
                minutes=int(runtime.get("timeout_minutes", 120))
            ),
        )
        return DistributedContext(True, rank, local_rank, env_world_size, device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(False, 0, 0, 1, device)


def wrap_model(model: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if not context.enabled:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def finalise_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def set_loader_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


__all__ = [
    "DistributedContext",
    "barrier",
    "finalise_distributed",
    "initialise_distributed",
    "set_loader_epoch",
    "unwrap_model",
    "wrap_model",
]
