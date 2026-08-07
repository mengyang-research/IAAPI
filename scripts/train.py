"""
Training script for IA-API model.

Usage:
    python scripts/train.py --config configs/default.yaml
    torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from iaapi.training.trainer import ISPTrainer


class AttrDict(dict):
    """Dictionary supporting attribute access for config compatibility."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({key: _to_attr_dict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(value) for value in obj]
    return obj


def load_config_file(path):
    """Load YAML config with a minimal Hydra-style defaults merge."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    merged = {}
    for entry in config.get("defaults", []) or []:
        default_name = entry if isinstance(entry, str) else next(iter(entry.values()))
        default_path = path.parent / f"{default_name}.yaml"
        if default_path.exists():
            merged = _deep_merge(merged, load_config_file(default_path))
    return _to_attr_dict(_deep_merge(merged, config))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train IA-API model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training when launched with torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    return rank, world_size, local_rank


def main():
    """Main training loop."""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config_file(config_path)
    if args.resume:
        config.logging.resume_from = args.resume
    if args.seed:
        config.system.seed = args.seed

    if config.system.distributed and config.system.num_gpus > 1:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        config.system.local_rank = local_rank
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = ISPTrainer(config, rank, world_size, device)
    trainer.train()

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
