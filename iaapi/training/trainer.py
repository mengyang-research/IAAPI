"""
ISP Trainer: Training framework for IA-API models.

Supports multi-GPU distributed training with mixed precision,
gradient accumulation, real sharded HDF5 datasets, and checkpointing.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except ImportError as exc:
    raise ImportError("PyTorch is required") from exc

try:  # WandB is optional when disabled in config.
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or OmegaConf-like config objects."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class ISPTrainer:
    """Trainer for IA-API models."""

    def __init__(
        self,
        config: Dict[str, Any],
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(_cfg(_cfg(config, "system", {}), "local_rank", rank))

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        if self.device.type == "cuda":
            device_index = self.device.index
            if device_index is None:
                device_index = self.local_rank if self.local_rank is not None else torch.cuda.current_device()
                self.device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device_index)

        seed = _cfg(_cfg(config, "system", {}), "seed", 42)
        self._set_seed(seed + rank)
        self.global_step = 0
        self.micro_step = 0
        self.best_metric = float("-inf")
        self.wandb_run = None

        self._init_components()

        self.checkpoint_dir = Path(_cfg(_cfg(config, "logging", {}), "checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._init_wandb()

        resume_from = _cfg(_cfg(config, "logging", {}), "resume_from", None)
        if resume_from:
            self._load_checkpoint(resume_from)

    def _init_wandb(self) -> None:
        """Initialize WandB only when enabled and available."""
        wandb_config = _cfg(_cfg(self.config, "logging", {}), "wandb", {})
        enabled = bool(_cfg(wandb_config, "enabled", True))
        if self.rank != 0 or not enabled:
            return
        if wandb is None:
            print("WandB is not installed; continuing without online logging")
            return
        self.wandb_run = wandb.init(
            project=_cfg(wandb_config, "project", "iaapi"),
            entity=_cfg(wandb_config, "entity", None),
            name=_cfg(wandb_config, "name", None),
            config=dict(self.config) if isinstance(self.config, dict) else None,
        )

    def _init_components(self):
        """Initialize model, optimizer, scheduler, scaler, and loss."""
        from iaapi.models.full_model import IAAPIModel
        from iaapi.training.losses import ISPLoss

        self.model = IAAPIModel(self.config).to(self.device)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            self.model = DDP(self.model, device_ids=device_ids)

        loss_config = _cfg(self.config, "loss", {})
        # log_std bounds for the width loss target clamp/mask; default to the
        # model's clamp range so the loss and the model agree on reachability.
        isp_cfg = _cfg(_cfg(self.config, "model", {}), "isp_head", {})
        default_min = _cfg(isp_cfg, "min_log_std", -7.0)
        default_max = _cfg(isp_cfg, "max_log_std", 5.0)
        self.loss_fn = ISPLoss(
            lambda_nll=_cfg(loss_config, "lambda_nll", 1.0),
            lambda_subspace=_cfg(loss_config, "lambda_subspace", 1.0),
            lambda_orth=_cfg(loss_config, "lambda_orth", 0.1),
            lambda_rank=_cfg(loss_config, "lambda_rank", 0.5),
            lambda_calibration=_cfg(loss_config, "lambda_calibration", 0.0),
            lambda_width=_cfg(loss_config, "lambda_width", 0.0),
            lambda_mean=_cfg(loss_config, "lambda_mean", 0.0),
            lambda_mean_regressor=_cfg(loss_config, "lambda_mean_regressor", 0.0),
            calibration_n_samples=_cfg(loss_config, "calibration_n_samples", 64),
            width_min_log_std=_cfg(loss_config, "width_min_log_std", default_min),
            width_max_log_std=_cfg(loss_config, "width_max_log_std", default_max),
            rank_class_balanced=_cfg(loss_config, "rank_class_balanced", False),
            rank_label_smoothing=_cfg(loss_config, "rank_label_smoothing", 0.0),
        )

        train_config = _cfg(self.config, "training", {})
        opt_config = _cfg(train_config, "optimizer", {})
        # C3 fix (root cause: rank gradient drowned by NLL): give the invariant
        # rank head its own optimizer with a higher learning rate so it learns
        # per-sample k from z_D instead of being out-competed by the NLL term.
        # The backbone (everything else) uses the standard AdamW below.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=_cfg(opt_config, "lr", 3e-4),
            betas=tuple(_cfg(opt_config, "betas", [0.9, 0.999])),
            weight_decay=_cfg(opt_config, "weight_decay", 1e-5),
        )
        rank_lr = _cfg(opt_config, "rank_lr", None)
        if rank_lr is not None:
            model_ref = self.model.module if isinstance(self.model, DDP) else self.model
            rank_params = []
            if hasattr(model_ref.isp_head, "rank_head"):
                rank_params = list(model_ref.isp_head.rank_head.parameters())
            if rank_params:
                rank_ids = {id(p) for p in rank_params}
                backbone_params = [p for p in self.model.parameters() if id(p) not in rank_ids]
                self.optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": _cfg(opt_config, "lr", 3e-4)},
                    {"params": rank_params, "lr": float(rank_lr)},
                ], betas=tuple(_cfg(opt_config, "betas", [0.9, 0.999])),
                   weight_decay=_cfg(opt_config, "weight_decay", 1e-5))
        self.scheduler = self._create_scheduler(
            self.optimizer,
            total_steps=_cfg(train_config, "max_steps", 200000),
            sched_config=_cfg(train_config, "scheduler", {}),
        )

        self.precision = str(_cfg(train_config, "precision", "bf16"))
        self.scaler = None
        if self.precision == "16" and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        sched_config: Dict[str, Any],
    ) -> torch.optim.lr_scheduler._LRScheduler:
        sched_name = _cfg(sched_config, "name", "cosine")
        if sched_name == "cosine":
            return CosineWarmupScheduler(
                optimizer,
                warmup_steps=_cfg(sched_config, "warmup_steps", 2000),
                max_steps=total_steps,
                min_lr=_cfg(sched_config, "min_lr", 1e-6),
            )
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    def _set_seed(self, seed: int) -> None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _autocast_context(self):
        """Return an autocast context appropriate for device/precision."""
        if self.precision == "32":
            return contextlib.nullcontext()
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        if self.precision == "bf16":
            return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def train(self):
        """Main training loop."""
        print(f"Starting training on rank {self.rank}")
        train_loader, val_loader = self._load_data()
        train_config = _cfg(self.config, "training", {})
        max_steps = _cfg(train_config, "max_steps", 200000)
        grad_accum = max(1, int(_cfg(train_config, "gradient_accumulation_steps", 1)))
        eval_config = _cfg(self.config, "evaluation", {})

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while self.global_step < max_steps:
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.global_step)
            n_batches = len(train_loader)
            for micro_idx, batch in enumerate(train_loader):
                batch = self._batch_to_device(batch)
                is_accum_boundary = ((micro_idx + 1) % grad_accum == 0)
                is_epoch_end = (micro_idx + 1 == n_batches)
                should_step = is_accum_boundary or is_epoch_end
                loss_dict = self._training_step(batch, should_step=should_step)
                if not should_step:
                    continue

                if self.rank == 0 and self.global_step % 100 == 0:
                    self._log(loss_dict)

                eval_every = _cfg(eval_config, "eval_every", 5000)
                if self.global_step > 0 and eval_every and self.global_step % eval_every == 0:
                    val_loss = self._validate(val_loader)
                    if self.rank == 0 and self.wandb_run is not None:
                        wandb.log({"val/loss": val_loss}, step=self.global_step)

                save_every = _cfg(eval_config, "save_every", 10000)
                if self.global_step > 0 and save_every and self.global_step % save_every == 0:
                    self._save_checkpoint()
                if self.global_step >= max_steps:
                    break
        self._save_checkpoint(final=True)

    def _training_step(self, batch: Dict[str, Any], should_step: bool = True) -> Dict[str, torch.Tensor]:
        """Run one microbatch; update optimizer only when should_step is true.

        Skips the optimizer step if the batch contains non-finite tensors
        (NaN/Inf trajectory or FIM eigenvalue) or if the resulting loss is
        non-finite. This stops a poisoned sample from cascading into the
        whole model.
        """
        grad_accum = max(1, int(_cfg(_cfg(self.config, "training", {}), "gradient_accumulation_steps", 1)))

        # Sanity check: drop batches with NaN/Inf inputs
        for key in ("theta_true", "fim_eigenvectors", "fim_eigenvalues"):
            v = batch.get(key)
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                # Skip this batch entirely
                zero = torch.zeros((), device=v.device, dtype=v.dtype)
                return {k: zero for k in ("total", "nll", "subspace", "orth", "rank", "calibration", "width")}
        obs = batch.get("observations", {})
        if isinstance(obs, dict):
            v = obs.get("values")
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                zero = torch.zeros((), device=v.device, dtype=v.dtype)
                return {k: zero for k in ("total", "nll", "subspace", "orth", "rank", "calibration", "width")}

        with self._autocast_context():
            outputs = self.model(batch)
            loss_dict = self.loss_fn(outputs, batch)
            scaled_loss = loss_dict["total"] / grad_accum

        # If loss is NaN/Inf, zero out grads and skip the step
        if not torch.isfinite(scaled_loss):
            self.optimizer.zero_grad(set_to_none=True)
            return {key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in loss_dict.items()}

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._after_optimizer_step()
        else:
            scaled_loss.backward()
            if should_step:
                self._clip_gradients()
                # Gradient NaN/Inf guard: skip the optimizer step if any gradient
                # is non-finite. Without this, a NaN gradient (common under bf16
                # autocast in the normalizing flow) corrupts the weights
                # permanently — every subsequent batch then produces NaN loss and
                # the run is stuck. Skipping the step preserves the last good weights.
                skip_step = False
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        skip_step = True
                        break
                if not skip_step:
                    self.optimizer.step()
                self._after_optimizer_step()

        return {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in loss_dict.items()}

    def _after_optimizer_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    def _clip_gradients(self) -> None:
        clip_norm = _cfg(_cfg(self.config, "training", {}), "clip_grad_norm", 1.0)
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

    def _validate(self, val_loader: Optional[DataLoader]) -> float:
        """Validation loop."""
        if val_loader is None or len(val_loader) == 0:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._batch_to_device(batch)
                with self._autocast_context():
                    outputs = self.model(batch)
                    loss_dict = self.loss_fn(outputs, batch)
                total_loss += float(loss_dict["total"].detach().cpu())
                n_batches += 1
        self.model.train()
        return total_loss / n_batches if n_batches else float("nan")

    def _load_data(self):
        """Load manifest-backed train and validation data."""
        from iaapi.data.data_generator import ShardedHDF5Dataset, collate_training_batch

        train_config = _cfg(self.config, "training", {})
        data_config = _cfg(self.config, "data", {})
        dataset_config = _cfg(data_config, "dataset", {})
        manifest_path = _cfg(dataset_config, "manifest_path", None)
        if manifest_path is None or not Path(manifest_path).exists():
            raise FileNotFoundError(
                f"Training manifest not found: {manifest_path}. "
                "Run scripts/generate_training_data.py first or set data.dataset.manifest_path."
            )

        train_dataset = ShardedHDF5Dataset(
            manifest_path=manifest_path,
            split="train",
            return_torch=True,
            reconstruct_graph=True,
        )
        val_dataset = ShardedHDF5Dataset(
            manifest_path=manifest_path,
            split="val",
            return_torch=True,
            reconstruct_graph=True,
        )
        batch_size = int(_cfg(train_config, "batch_size", 64))
        num_workers = int(_cfg(train_config, "num_workers", 0))
        pin_memory = self.device.type == "cuda"

        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_training_batch,
        )
        val_loader = None
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_training_batch,
            )
        return train_loader, val_loader

    def _batch_to_device(self, batch: Any) -> Any:
        """Recursively move tensors and PyG batches to the trainer device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {key: self._batch_to_device(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._batch_to_device(value) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self._batch_to_device(value) for value in batch)
        if hasattr(batch, "to") and batch.__class__.__module__.startswith("torch_geometric"):
            return batch.to(self.device)
        return batch

    def _log(self, loss_dict: Dict[str, Any]) -> None:
        """Log training metrics."""
        if self.rank != 0:
            return
        lr = self.optimizer.param_groups[0]["lr"]
        log_dict = {"train/step": self.global_step, "train/lr": lr}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                log_dict[f"train/{key if key != 'total' else 'loss'}"] = float(value.detach().cpu())
            else:
                log_dict[f"train/{key if key != 'total' else 'loss'}"] = value
        if self.wandb_run is not None:
            wandb.log(log_dict, step=self.global_step)
        else:
            print(log_dict)

    def _model_state_dict(self) -> Dict[str, Any]:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return model.state_dict()

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        state = {
            "state_dict": self._model_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(state, checkpoint_path)
        if final:
            final_path = self.checkpoint_dir / "final_model.pt"
            torch.save(state, final_path)
            print(f"Saved final checkpoint to {final_path}")
        else:
            print(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model/optimizer/scheduler/scaler state from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        state_dict = checkpoint["state_dict"]
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine learning rate schedule with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        return [lr for _ in self.optimizer.param_groups]
