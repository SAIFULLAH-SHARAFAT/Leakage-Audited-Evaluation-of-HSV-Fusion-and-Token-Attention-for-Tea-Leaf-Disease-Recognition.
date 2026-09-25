import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re
import csv
import json
import time
import math
import shutil
import random
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, ContextManager, Union
from collections import Counter
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

from training_contracts import (
    matched_initialization, rng_neutral, EpochDataset, start_epoch,
    optimizer_groups, accumulation_weight, optimizer_update, ema_evaluation,
    immutable_config, checkpoint_contract, resume_training, selection_metadata,
    validate_eval_index,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

import timm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            warnings.warn(
                f"Could not enable deterministic algorithms: {e}\n"
                "Training will continue with partial determinism."
            )
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_autocast_ctx(use_amp: bool) -> ContextManager:
    enabled = bool(use_amp and torch.cuda.is_available())
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(use_amp: bool):
    enabled = bool(use_amp and torch.cuda.is_available())
    try:
        return torch.cuda.amp.GradScaler(enabled=enabled)
    except Exception:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(enabled=enabled)
        raise


def rgb_to_hsv_torch(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    maxc, _ = rgb01.max(dim=1, keepdim=True)
    minc, _ = rgb01.min(dim=1, keepdim=True)
    v = maxc
    delta = maxc - minc
    
    eps = 1e-10
    
    # Saturation
    s = delta / torch.clamp(maxc, min=eps)
    s = torch.where(maxc > eps, s, torch.zeros_like(s))
    
    # Hue computation with deterministic max-channel selection
    delta_safe = torch.clamp(delta, min=eps)
    
    # Compute hue formulas for each channel (standard HSV conversion)
    h_r = (g - b) / delta_safe          # If R is max
    h_g = (b - r) / delta_safe + 2.0    # If G is max
    h_b = (r - g) / delta_safe + 4.0    # If B is max
    
    idx = rgb01.argmax(dim=1, keepdim=True)   # [B,1,H,W] in {0,1,2}
    mask = (delta > eps)                      # [B,1,H,W]

    h = torch.zeros_like(delta, dtype=rgb01.dtype)  # [B,1,H,W]
    h = torch.where((idx == 0) & mask, h_r, h)
    h = torch.where((idx == 1) & mask, h_g, h)
    h = torch.where((idx == 2) & mask, h_b, h)

    # Normalize to [0, 1) with true modulo (handles wraparound correctly)
    h = torch.remainder(h / 6.0, 1.0)
    hsv = torch.cat([h, s, v], dim=1)
    return torch.clamp(hsv, 0.0, 1.0)


def hsv_to_sincos_sv(hsv01: torch.Tensor) -> torch.Tensor:
    """
    HSV [0,1] -> [sin(2*pi*H), cos(2*pi*H), S, V]
    Removes hue discontinuity at wrap-around.
    """
    h = hsv01[:, 0:1]
    s = hsv01[:, 1:2]
    v = hsv01[:, 2:3]
    ang = 2.0 * math.pi * h
    hsin = torch.sin(ang)
    hcos = torch.cos(ang)
    return torch.cat([hsin, hcos, s, v], dim=1)


def normalize_hsv_rep(x: torch.Tensor) -> torch.Tensor:
    """
    x: [hsin, hcos, s, v]
    Keep sin/cos as-is in [-1,1].
    Normalize S,V to roughly zero-mean and stable scale.
    """
    hsin = x[:, 0:1]
    hcos = x[:, 1:2]
    s = (x[:, 2:3] - 0.5) / 0.25
    v = (x[:, 3:4] - 0.5) / 0.25
    return torch.cat([hsin, hcos, s, v], dim=1)


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def try_get_gflops(model: nn.Module, input_size: int, device: str) -> Optional[float]:
    """GFLOPs via ptflops (optional). Returns None if unavailable."""
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None

    if isinstance(model, nn.DataParallel):
        model = model.module

    model = model.to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            macs, _params = get_model_complexity_info(
                model, (3, input_size, input_size),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
        flops = 2.0 * float(macs)  # FLOPs ≈ 2*MACs
        return flops / 1e9
    except Exception as e:
        print(f"GFLOPs computation failed: {e}")
        return None
    finally:
        if was_training:
            model.train()


def _json_safe_scalar(x):
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return _json_safe_scalar(x.item())
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _json_safe_scalar(x.item())
        return x.tolist()
    return str(x)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        out[k if k.startswith("module.") else f"module.{k}"] = v
    return out


def load_state_dict_robust(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Robust load:
    - strict=True with (as-is / strip module / add module)
    - complete key/shape coverage required before mutating the model
    """
    prefix_fns = [lambda x: x, strip_module_prefix, add_module_prefix]
    expected = model.state_dict()
    for fn in prefix_fns:
        candidate = fn(state_dict)
        if set(candidate) == set(expected) and all(
                candidate[k].shape == expected[k].shape for k in expected):
            model.load_state_dict(candidate, strict=True)
            return

    raise RuntimeError("Could not load state dict with any strategy.")


# -------------------------
# EMA
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = p.data.clone()
                else:
                    self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        if self.backup:
            raise RuntimeError("EMA shadow is already installed")
        self.backup = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name].clone()

    def restore(self):
        if not self.backup:
            warnings.warn("EMA.restore() called but backup is empty. Did you call apply_shadow()?")
            return
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data = self.backup[name].clone()
        self.backup = {}


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    campaign_id: str = "tea_leaf_v2"
    # Model
    model_name: str = "swin_tiny_patch4_window7_224"
    num_classes: int = 7
    pretrained: bool = False
    drop_path_rate: float = 0.2

    # Data
    data_root: str = "/kaggle/input/tea-leaf701515/tea_leaf_processed_dataset"
    input_size: int = 224
    batch_size: int = 64
    num_workers: int = 2

    # ImageNet stats
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # HSV stabilization
    hsv_use_sincos: bool = True     # using sin/cos hue representation
    gate_warmup_epochs: int = 5     # gate_alpha ramps 0->1 over the epochs

    # Training
    epochs: int = 80
    warmup_epochs: int = 5
    lr: float = 5e-4
    warmup_lr_init: float = 1e-6
    weight_decay: float = 0.05
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Augmentation
    color_jitter: float = 0.2
    hue_jitter: float = 0.0
    use_gaussian_blur: bool = False
    gaussian_blur_prob: float = 0.1
    rrc_scale_min: float = 0.85

    # Optimization
    use_amp: bool = True
    use_ema: bool = False
    ema_decay: float = 0.9998
    use_class_weights: bool = False

    # HSV branch
    use_hsv_branch: bool = False
    hsv_embed_dim: int = 128
    hsv_dropout: float = 0.1
    fuse_dropout: float = 0.2
    gate_hidden: int = 256
    gate_vector: bool = False

    # Mechanistic diagnostic (no optimization/state change; HSV variants only)
    log_init_scale_diagnostics: bool = False

    # Evaluation
    compute_val_auc: bool = False
    early_stopping_patience: int = 25

    # Logging / folders
    run_dir: str = "/kaggle/working/experiments_swin"
    experiment_name: str = "rgb_baseline"
    log_interval: int = 50
    save_cm_png: bool = True

    # Checkpointing
    ckpt_temp_dir: str = "/kaggle/temp"
    save_epoch_checkpoints: bool = False
    save_epoch_every: int = 5
    keep_last_n_checkpoints: int = 3

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    deterministic: bool = True
    use_data_parallel: bool = True

    def validate(self):
        invalid_chars = ['/', '\\', '..', '\0']
        if any(char in self.experiment_name for char in invalid_chars):
            raise ValueError(f"Invalid experiment_name: {self.experiment_name}")
        if not Path(self.data_root).exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0,1)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 <= self.ema_decay <= 1.0):
            raise ValueError("ema_decay must be in [0,1]")
        if self.drop_path_rate < 0:
            raise ValueError("drop_path_rate must be >= 0")


# -------------------------
# Model
# -------------------------
class SwinRGBHSV(nn.Module):
    @matched_initialization
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_path_rate: float,
        num_classes: int,
        use_hsv_branch: bool,
        hsv_embed_dim: int = 128,
        hsv_use_sincos: bool = True,
        hsv_dropout: float = 0.1,
        fuse_dropout: float = 0.2,
        gate_hidden: int = 256,
        gate_vector: bool = False,
        img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.use_hsv_branch = use_hsv_branch
        self.hsv_use_sincos = hsv_use_sincos
        self.gate_vector = gate_vector

        self.register_buffer("img_mean", torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(img_std).view(1, 3, 1, 1))

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            global_pool="avg",
        )
        feat_dim = getattr(self.backbone, "num_features", 768)
        self.feat_dim = int(feat_dim)

        if self.use_hsv_branch:
            in_ch = 4 if self.hsv_use_sincos else 3

            self.hsv_branch = nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=hsv_dropout),
                nn.Linear(32, hsv_embed_dim),
                nn.GELU(),
            )

            self.hsv_proj = nn.Sequential(
                nn.Dropout(p=fuse_dropout),
                nn.Linear(hsv_embed_dim, self.feat_dim),
            )

            gate_out_dim = self.feat_dim if self.gate_vector else 1
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.feat_dim + hsv_embed_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(p=fuse_dropout),
                nn.Linear(gate_hidden, gate_out_dim),
            )
            if self.gate_mlp[-1].bias is not None:
                nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

        self.classifier = nn.Sequential(
            nn.Dropout(p=fuse_dropout),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.GELU(),
            nn.Dropout(p=fuse_dropout),
            nn.Linear(self.feat_dim // 2, num_classes),
        )

    def forward(self, x_norm: torch.Tensor, return_gate: bool = False, gate_alpha: float = 1.0):
        feat = self.backbone(x_norm)

        if not self.use_hsv_branch:
            logits = self.classifier(feat)
            if return_gate:
                gate = torch.zeros((logits.size(0), 1), device=logits.device, dtype=logits.dtype)
                return logits, gate
            return logits

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=False):
                x_rgb01 = (x_norm.float() * self.img_std.float()) + self.img_mean.float()
                x_rgb01 = torch.clamp(x_rgb01, 0.0, 1.0)

                x_hsv = rgb_to_hsv_torch(x_rgb01)
                if self.hsv_use_sincos:
                    x_hsv_rep = hsv_to_sincos_sv(x_hsv)
                    x_hsv_rep = normalize_hsv_rep(x_hsv_rep)
                else:
                    x_hsv_rep = x_hsv
        else:
            x_rgb01 = (x_norm.float() * self.img_std.float()) + self.img_mean.float()
            x_rgb01 = torch.clamp(x_rgb01, 0.0, 1.0)

            x_hsv = rgb_to_hsv_torch(x_rgb01)
            if self.hsv_use_sincos:
                x_hsv_rep = hsv_to_sincos_sv(x_hsv)
                x_hsv_rep = normalize_hsv_rep(x_hsv_rep)
            else:
                x_hsv_rep = x_hsv

        x_hsv_rep = x_hsv_rep.to(dtype=feat.dtype)
        hsv_emb = self.hsv_branch(x_hsv_rep)

        gate_in = torch.cat([feat.float(), hsv_emb.float()], dim=1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))

        if not self.gate_vector:
            gate_for_fuse = gate.expand(-1, self.feat_dim)
            gate_stat = gate
        else:
            gate_for_fuse = gate
            gate_stat = gate.mean(dim=1, keepdim=True)

        hsv_feat = self.hsv_proj(hsv_emb)

        gate_for_fuse = gate_for_fuse.to(dtype=feat.dtype)
        hsv_feat = hsv_feat.to(dtype=feat.dtype)

        alpha = float(gate_alpha)
        fused = feat + (alpha * gate_for_fuse) * hsv_feat
        logits = self.classifier(fused)

        if return_gate:
            return logits, gate_stat
        return logits

    @torch.no_grad()
    def initialization_scale_diagnostics(
        self,
        x_norm: torch.Tensor,
        first_epoch_gate_alpha: float = 0.2,
    ) -> Dict[str, float]:
        if not self.use_hsv_branch:
            raise RuntimeError("HSV scale diagnostics require use_hsv_branch=True")

        def rms(z: torch.Tensor) -> torch.Tensor:
            z = z.float()
            return z.square().mean().sqrt()

        feat = self.backbone(x_norm)
        x_rgb01 = (x_norm.float() * self.img_std.float()) + self.img_mean.float()
        x_rgb01 = torch.clamp(x_rgb01, 0.0, 1.0)
        x_hsv = rgb_to_hsv_torch(x_rgb01)
        if self.hsv_use_sincos:
            x_hsv_rep = normalize_hsv_rep(hsv_to_sincos_sv(x_hsv))
        else:
            x_hsv_rep = x_hsv
        x_hsv_rep = x_hsv_rep.to(dtype=feat.dtype)

        hsv_emb = self.hsv_branch(x_hsv_rep)
        hsv_feat = self.hsv_proj(hsv_emb)
        gate_in = torch.cat([feat.float(), hsv_emb.float()], dim=1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))
        gate_for_fuse = gate if self.gate_vector else gate.expand(-1, self.feat_dim)

        feat32 = feat.float()
        hsv32 = hsv_feat.float()
        gate32 = gate_for_fuse.float()
        full_delta = gate32 * hsv32
        epoch1_delta = float(first_epoch_gate_alpha) * full_delta
        fused_full = feat32 + full_delta

        logits_rgb = self.classifier(feat32)
        logits_fused = self.classifier(fused_full)

        rgb_rms = rms(feat32)
        hsv_rms = rms(hsv32)
        delta_rms = rms(full_delta)
        epoch1_delta_rms = rms(epoch1_delta)
        logit_delta_rms = rms(logits_fused.float() - logits_rgb.float())
        eps = 1e-12

        cos = F.cosine_similarity(feat32, hsv32, dim=1).mean()
        return {
            "n_images": int(x_norm.shape[0]),
            "rgb_feature_rms": float(rgb_rms.item()),
            "hsv_embedding_rms": float(rms(hsv_emb).item()),
            "hsv_projected_rms": float(hsv_rms.item()),
            "hsv_projected_to_rgb_rms": float((hsv_rms / (rgb_rms + eps)).item()),
            "gate_mean": float(gate.float().mean().item()),
            "gate_std": float(gate.float().std(unbiased=False).item()),
            "gate_min": float(gate.float().min().item()),
            "gate_max": float(gate.float().max().item()),
            "full_gate_delta_rms": float(delta_rms.item()),
            "full_gate_delta_to_rgb_rms": float((delta_rms / (rgb_rms + eps)).item()),
            "first_epoch_gate_alpha": float(first_epoch_gate_alpha),
            "first_epoch_delta_rms": float(epoch1_delta_rms.item()),
            "first_epoch_delta_to_rgb_rms": float((epoch1_delta_rms / (rgb_rms + eps)).item()),
            "feature_cosine_mean": float(cos.item()),
            "full_gate_logit_delta_rms": float(logit_delta_rms.item()),
            "interpretation_note": (
                "Diagnostic only. A small gated contribution can be intentional because "
                "the gate is conservatively initialized. The key scale check is the "
                "ungated hsv_projected_to_rgb_rms ratio."
            ),
        }


# -------------------------
# Checkpoint Manager
# -------------------------
class CheckpointManager:
    def __init__(self, temp_root: Path, final_root: Path, experiment_name: str):
        self.temp_dir = temp_root / experiment_name
        self.final_dir = final_root / experiment_name
        self.pid = os.getpid()

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        (self.final_dir / "model").mkdir(parents=True, exist_ok=True)
        (self.final_dir / "config").mkdir(parents=True, exist_ok=True)
        (self.final_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.final_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.final_dir / "raw_outputs").mkdir(parents=True, exist_ok=True)
        (self.final_dir / "figures").mkdir(parents=True, exist_ok=True)

        print(f"\n Directories:")
        print(f"   Temp:  {self.temp_dir}")
        print(f"   Final: {self.final_dir}")

    def save_checkpoint(
        self,
        checkpoint: Dict,
        epoch: int,
        is_best: bool,
        save_epoch_ckpt: bool = False,
        epoch_interval: int = 5,
        keep_n: int = 3
    ) -> None:
        def _atomic_save(obj, final_path: Path) -> bool:
            tmp_path = final_path.with_suffix(f".{self.pid}.tmp")
            try:
                torch.save(obj, tmp_path)
                tmp_path.replace(final_path)
                return True
            except Exception as e:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                print(f"Checkpoint save failed ({final_path.name}): {repr(e)}")
                return False

        if not _atomic_save(checkpoint, self.temp_dir / "last_checkpoint.pth"):
            raise IOError("Required last checkpoint could not be saved")
        if is_best:
            if not _atomic_save(checkpoint, self.temp_dir / "best_model.pth"):
                raise IOError("Required best checkpoint could not be saved")

        if save_epoch_ckpt and (not is_best) and (epoch % max(1, epoch_interval) == 0):
            ckpt_path = self.temp_dir / f"checkpoint_epoch_{epoch}.pth"
            if _atomic_save(checkpoint, ckpt_path):
                self._cleanup_old_epochs(keep_n)

    def _cleanup_old_epochs(self, keep_n: int) -> None:
        epoch_ckpts = sorted(
            self.temp_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: int(re.search(r"checkpoint_epoch_(\d+)", p.name).group(1))
            if re.search(r"checkpoint_epoch_(\d+)", p.name) else 0
        )
        if len(epoch_ckpts) <= keep_n:
            return
        for old in epoch_ckpts[:-keep_n]:
            try:
                old.unlink()
            except Exception:
                pass

    def _save_confusion_matrix_png(self, cm: np.ndarray, classes: List[str], out_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(cm, interpolation="nearest")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(len(classes)),
                yticks=np.arange(len(classes)),
                xticklabels=classes,
                yticklabels=classes,
                ylabel="True label",
                xlabel="Predicted label",
                title="Confusion Matrix"
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Could not save confusion_matrix.png: {e}")

    def copy_final_artifacts(
        self,
        best_ckpt_path: Path,
        config: Config,
        train_log: List[Dict],
        test_metrics: Dict,
        class_to_idx: Dict[str, int],
        classes: List[str],
        logs_dir: Path,
    ) -> None:
        print("\n" + "=" * 80)
        print("COPYING FINAL ARTIFACTS")
        print("=" * 80)

        model_dir = self.final_dir / "model"
        config_dir = self.final_dir / "config"
        logs_dst = self.final_dir / "logs"
        metrics_dir = self.final_dir / "metrics"
        raw_dir = self.final_dir / "raw_outputs"
        fig_dir = self.final_dir / "figures"

        if not best_ckpt_path.exists():
            raise FileNotFoundError(f"best_model.pth not found: {best_ckpt_path}")

        shutil.copy2(best_ckpt_path, model_dir / "best_model.pth")
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)

        torch.save(best_ckpt["model"], model_dir / "model_weights_only.pth")

        if "ema_shadow" in best_ckpt and isinstance(best_ckpt["ema_shadow"], dict):
            ema_state = dict(best_ckpt["model"])
            for k, v in best_ckpt["ema_shadow"].items():
                if k in ema_state:
                    ema_state[k] = v
            torch.save(ema_state, model_dir / "ema_model_weights_only.pth")

        with open(config_dir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
        with open(config_dir / "class_to_idx.json", "w") as f:
            json.dump(class_to_idx, f, indent=2)
        with open(config_dir / "classes.json", "w") as f:
            json.dump(classes, f, indent=2)

        if train_log:
            with open(logs_dst / "train_log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
                writer.writeheader()
                writer.writerows(train_log)

        gate_src = Path(logs_dir) / "gate_stats_log.csv"
        gate_dst = logs_dst / "gate_stats_log.csv"
        if gate_src.exists():
            if gate_src.resolve() != gate_dst.resolve():
                shutil.copy2(gate_src, gate_dst)

        results = {}
        for k, v in test_metrics.items():
            if not isinstance(v, (np.ndarray, list, dict)):
                scalar_val = _json_safe_scalar(v)
                if k == "macro_auc" and scalar_val == -1.0:
                    results[k] = "N/A"
                else:
                    results[k] = scalar_val

        with open(metrics_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        if "targets" in test_metrics and "preds" in test_metrics:
            cm = confusion_matrix(test_metrics["targets"], test_metrics["preds"])
            np.save(metrics_dir / "confusion_matrix.npy", cm)

            rep = classification_report(
                test_metrics["targets"],
                test_metrics["preds"],
                target_names=classes,
                digits=4
            )
            with open(metrics_dir / "classification_report.txt", "w") as f:
                f.write(rep)

            per_class_f1 = f1_score(
                test_metrics["targets"], test_metrics["preds"],
                average=None, labels=list(range(len(classes)))
            )
            per_class_precision = precision_score(
                test_metrics["targets"], test_metrics["preds"],
                average=None, labels=list(range(len(classes))), zero_division=0
            )
            per_class_recall = recall_score(
                test_metrics["targets"], test_metrics["preds"],
                average=None, labels=list(range(len(classes))), zero_division=0
            )

            per_class_metrics = {}
            for i, cname in enumerate(classes):
                per_class_metrics[cname] = {
                    "f1": float(per_class_f1[i]),
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                }
            with open(metrics_dir / "per_class_metrics.json", "w") as f:
                json.dump(per_class_metrics, f, indent=2)

            if config.save_cm_png:
                self._save_confusion_matrix_png(cm, classes, fig_dir / "confusion_matrix.png")

        if "preds" in test_metrics:
            np.save(raw_dir / "test_predictions.npy", test_metrics["preds"])
        if "targets" in test_metrics:
            np.save(raw_dir / "test_targets.npy", test_metrics["targets"])
        if "probs" in test_metrics:
            np.save(raw_dir / "test_probabilities.npy", test_metrics["probs"])
        if "sample_ids" in test_metrics:
            ids = [str(x) for x in test_metrics["sample_ids"]]
            if "targets" in test_metrics and len(ids) != len(test_metrics["targets"]):
                raise ValueError("sample_ids length does not match test targets")
            if len(ids) != len(set(ids)):
                raise ValueError("test sample_ids are not unique")
            with open(raw_dir / "test_ids.json", "w") as f:
                json.dump(ids, f, indent=2)
        if "gate_stats" in test_metrics:
            with open(raw_dir / "gate_stats_test.json", "w") as f:
                json.dump(test_metrics["gate_stats"], f, indent=2)

        # Atomic completion marker
        complete_payload = {
            "status": "complete",
            "campaign_id": getattr(config, "campaign_id", None),
            "experiment_name": getattr(config, "experiment_name", None),
            "seed": getattr(config, "seed", None),
            "n_test": int(len(test_metrics["targets"])) if "targets" in test_metrics else None,
        }
        tmp = self.final_dir / f"COMPLETE.{self.pid}.tmp"
        dst = self.final_dir / "COMPLETE.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(complete_payload, f, indent=2)
        tmp.replace(dst)

        print(f"Final bundle ready at: {self.final_dir}")


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        set_seed(cfg.seed, deterministic=cfg.deterministic)

        if cfg.deterministic and cfg.num_workers > 0:
            print(
                "Determinism note: num_workers > 0 with random augmentations may still introduce "
                "small non-determinism. For maximum determinism, set --num_workers 0."
            )

        self.run_root = Path(cfg.run_dir)
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.exp_dir = self.run_root / cfg.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.exp_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.config_dir = self.exp_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        self.ckpt_manager = CheckpointManager(
            temp_root=Path(cfg.ckpt_temp_dir),
            final_root=self.run_root,
            experiment_name=cfg.experiment_name
        )

        self.model = self._build_model().to(cfg.device)

        self.params_m = float(count_params_m(self.model))
        self.gflops = try_get_gflops(self.model, cfg.input_size, cfg.device)
        self.gflops_out = float(self.gflops) if self.gflops is not None else -1.0

        print(f"   Params: {self.params_m:.2f}M")
        print(f"   GFLOPs: {self.gflops_out:.2f}" if self.gflops_out >= 0 else "   GFLOPs: N/A")

        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1 and cfg.use_data_parallel:
            print(f"\n DataParallel: {self.n_gpus} GPUs")
            self.model = nn.DataParallel(self.model)
        else:
            if torch.cuda.is_available():
                print(f"\n Single GPU: {torch.cuda.get_device_name(0)}")
            self.n_gpus = 1

        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.ema = EMA(base_model, cfg.ema_decay) if cfg.use_ema else None

        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()

        self.class_weights = None
        if cfg.use_class_weights:
            class_counts = Counter(self.train_loader.dataset.targets)
            weights = []
            missing = []
            for i in range(cfg.num_classes):
                c = int(class_counts.get(i, 0))
                if c == 0:
                    missing.append(i)
                weights.append(1.0 / max(c, 1))

            if missing:
                warnings.warn(
                    f"Some classes are missing in TRAIN split: {missing}. "
                    "Using weight=1.0 for them."
                )

            mean_w = sum(weights) / max(1, len(weights))
            weights = [w / max(mean_w, 1e-12) for w in weights]

            self.class_weights = torch.tensor(weights, dtype=torch.float32, device=cfg.device)
            print(f"\n  Using class weights: {[f'{w:.3f}' for w in weights]}")

        self.optimizer = torch.optim.AdamW(
            optimizer_groups(self.model, cfg.weight_decay),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        batches_per_epoch = len(self.train_loader)
        accum = int(cfg.gradient_accumulation_steps)
        updates_per_epoch = int(math.ceil(batches_per_epoch / accum))
        total_updates = int(cfg.epochs * updates_per_epoch)
        warmup_updates = int(cfg.warmup_epochs * updates_per_epoch)
        warmup_updates = min(warmup_updates, max(0, total_updates - 1))

        print("\n Scheduler Setup (optimizer updates):")
        print(f"   batches/epoch: {batches_per_epoch}")
        print(f"   accum_steps:   {accum}")
        print(f"   updates/epoch: {updates_per_epoch}")
        print(f"   warmup_updates:{warmup_updates}")
        print(f"   total_updates: {total_updates}")

        if warmup_updates > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=cfg.warmup_lr_init / cfg.lr,
                end_factor=1.0,
                total_iters=warmup_updates,
            )
            cosine_updates = max(1, total_updates - warmup_updates)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_updates,
                eta_min=cfg.min_lr,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_updates],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_updates),
                eta_min=cfg.min_lr,
            )

        self.scaler = make_grad_scaler(cfg.use_amp)

        self.best_val_f1 = -1.0
        self.bad_epochs = 0
        self.start_epoch = 1
        self.train_log: List[Dict[str, float]] = []

        self._save_config()
        self._save_env()
        self._validate_dataset()

    def _save_config(self):
        immutable_config(self.config_dir, self.cfg)

    def _save_env(self):
        if (self.config_dir / "env.json").exists():
            return
        info = {
            "python": os.popen("python -V").read().strip(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        with open(self.config_dir / "env.json", "w") as f:
            json.dump(info, f, indent=2)

    def _validate_dataset(self):
        print("\n Dataset Validation:")
        train_targets = self.train_loader.dataset.targets
        val_targets = self.val_loader.dataset.targets
        test_targets = self.test_loader.dataset.targets

        train_dist = Counter(train_targets)
        val_dist = Counter(val_targets)
        test_dist = Counter(test_targets)

        class_names = self.train_loader.dataset.classes
        print("\n  Class Distribution:")
        counts = []
        for ci, cname in enumerate(class_names):
            tr = train_dist.get(ci, 0)
            va = val_dist.get(ci, 0)
            te = test_dist.get(ci, 0)
            print(f"    {cname:20s} -> Train: {tr:5d} | Val: {va:5d} | Test: {te:5d}")
            counts.append(tr)
            if tr < 10:
                print(f"Very low train samples for class '{cname}': {tr}")

        if len(counts) > 0 and min(counts) > 0:
            ratio = max(counts) / min(counts)
            if ratio > 10:
                print(f"\n Severe imbalance ratio: {ratio:.1f}:1 (max/min train class)")
                print(f"      Consider: class weighting, focal loss, or resampling")
            elif ratio > 3:
                print(f"\n  Moderate imbalance ratio: {ratio:.1f}:1")

            median_count = np.median(counts)
            for ci, cname in enumerate(class_names):
                if counts[ci] < median_count * 0.5:
                    print(
                        f" Class '{cname}' underrepresented: "
                        f"{counts[ci]} samples (median: {int(median_count)})"
                    )
        print()

    @rng_neutral
    def _write_initialization_scale_diagnostics(self) -> None:
        """Write one pre-optimization scale diagnostic from the fixed val loader."""
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        was_training = base_model.training
        base_model.eval()
        try:
            batch = next(iter(self.val_loader))
            x = batch[0].to(self.cfg.device, non_blocking=True)
            first_alpha = 1.0 / max(1, int(self.cfg.gate_warmup_epochs))
            diag = base_model.initialization_scale_diagnostics(
                x, first_epoch_gate_alpha=first_alpha
            )
            diag.update({
                "experiment_name": self.cfg.experiment_name,
                "seed": int(self.cfg.seed),
                "hsv_use_sincos": bool(self.cfg.hsv_use_sincos),
                "gate_vector": bool(self.cfg.gate_vector),
                "measured_before_optimizer_updates": True,
                "data_split": "validation",
            })
            out = self.ckpt_manager.final_dir / "metrics" / "init_scale_diagnostics.json"
            with open(out, "w") as f:
                json.dump(diag, f, indent=2)
            print("\nInitialization-scale diagnostic:")
            print(json.dumps(diag, indent=2))
            print(f"   wrote: {out}")
        finally:
            base_model.train(was_training)

    def _build_model(self) -> nn.Module:
        print(f"\nBuilding model: {self.cfg.model_name}")
        print(f"   Pretrained: {self.cfg.pretrained}")
        print(f"   DropPath:   {self.cfg.drop_path_rate}")
        print(f"   RGB+HSV:    {self.cfg.use_hsv_branch} (gate_vector={self.cfg.gate_vector})")
        print(f"   HSV sincos: {self.cfg.hsv_use_sincos} | gate warmup: {self.cfg.gate_warmup_epochs} ep")

        return SwinRGBHSV(
            model_name=self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            drop_path_rate=self.cfg.drop_path_rate,
            num_classes=self.cfg.num_classes,
            use_hsv_branch=self.cfg.use_hsv_branch,
            hsv_embed_dim=self.cfg.hsv_embed_dim,
            hsv_use_sincos=self.cfg.hsv_use_sincos,
            hsv_dropout=self.cfg.hsv_dropout,
            fuse_dropout=self.cfg.fuse_dropout,
            gate_hidden=self.cfg.gate_hidden,
            gate_vector=self.cfg.gate_vector,
            img_mean=self.cfg.img_mean,
            img_std=self.cfg.img_std,
        )

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        root = Path(self.cfg.data_root)
        for split in ["train", "val", "test"]:
            if not (root / split).exists():
                raise FileNotFoundError(f"Missing split folder: {root/split}")

        hue_for_train = float(self.cfg.hue_jitter)

        train_tf_list = [
            T.RandomResizedCrop(
                self.cfg.input_size,
                scale=(self.cfg.rrc_scale_min, 1.0),
                interpolation=InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=self.cfg.color_jitter,
                contrast=self.cfg.color_jitter,
                saturation=self.cfg.color_jitter,
                hue=hue_for_train
            ),
        ]
        if self.cfg.use_gaussian_blur:
            train_tf_list.append(
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=self.cfg.gaussian_blur_prob)
            )
        train_tf_list += [
            T.ToTensor(),
            T.Normalize(mean=self.cfg.img_mean, std=self.cfg.img_std),
        ]
        train_tf = T.Compose(train_tf_list)

        eval_tf = T.Compose([
            T.Resize(int(self.cfg.input_size * 1.14), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(self.cfg.input_size),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.img_mean, std=self.cfg.img_std),
        ])

        train_ds = EpochDataset(ImageFolder(root / "train", transform=train_tf), self.cfg.seed)
        val_ds = ImageFolder(root / "val", transform=eval_tf)
        test_ds = ImageFolder(root / "test", transform=eval_tf)
        validate_eval_index(val_ds, "val", root)
        validate_eval_index(test_ds, "test", root)

        if len(train_ds.classes) != self.cfg.num_classes:
            raise ValueError(
                f"num_classes={self.cfg.num_classes} but dataset has {len(train_ds.classes)} classes: {train_ds.classes}"
            )

        print(f"\nData: Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
        print(f"   Classes: {train_ds.classes}")
        print(f"   Hue jitter: {self.cfg.hue_jitter} (identical policy across R0-R3)")

        use_persistent = False

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            persistent_workers=use_persistent,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )
        return train_loader, val_loader, test_loader

    def _append_gate_csv(self, epoch: int, gate_stats: Dict[str, Dict[str, float]], gate_global_mean: float):
        gate_csv_path = self.logs_dir / "gate_stats_log.csv"
        row: Dict[str, Union[int, float]] = {"epoch": int(epoch), "gate_global_mean": float(gate_global_mean)}
        for cname, stats in gate_stats.items():
            row[f"{cname}_mean"] = float(stats["mean"])
            row[f"{cname}_std"] = float(stats["std"])
            row[f"{cname}_n"] = int(stats.get("n", 0))

        write_header = not gate_csv_path.exists()
        with open(gate_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def resume_from(self, ckpt_path: Path) -> None:
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        resume_training(self, ckpt_path, model, load_state_dict_robust)

    def _gate_alpha_for_epoch(self, epoch: int) -> float:
        if not self.cfg.use_hsv_branch:
            return 1.0
        w = max(1, int(self.cfg.gate_warmup_epochs))
        return float(min(1.0, epoch / w))

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        start_epoch(self.train_loader, epoch, self.cfg.seed)
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        accum_steps = int(self.cfg.gradient_accumulation_steps)

        gate_alpha = self._gate_alpha_for_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.cfg.epochs}", leave=False)

        for i, (x, y) in enumerate(pbar, start=1):
            x = x.to(self.cfg.device, non_blocking=True)
            y = y.to(self.cfg.device, non_blocking=True)

            with get_autocast_ctx(self.cfg.use_amp):
                logits = self.model(x, gate_alpha=gate_alpha)
                loss = F.cross_entropy(
                    logits, y,
                    weight=self.class_weights,
                    label_smoothing=self.cfg.label_smoothing
                )

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

            self.scaler.scale(loss * accumulation_weight(
                self.train_loader, i, accum_steps, x.size(0))).backward()

            do_step = (i % accum_steps == 0) or (i == len(self.train_loader))
            if do_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

                optimizer_update(self)

                self.optimizer.zero_grad(set_to_none=True)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

            if i % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}", "gate_a": f"{gate_alpha:.2f}"})

        return {"loss": total_loss / max(1, n), "acc1": correct / max(1, n)}

    def _compute_gate_stats(
        self,
        gates_1d: np.ndarray,
        targets: np.ndarray,
        class_names: List[str],
        class_to_idx: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for cname in class_names:
            ci = class_to_idx[cname]
            idx = (targets == ci)
            if idx.sum() == 0:
                out[cname] = {"mean": 0.0, "std": 0.0, "n": 0}
            else:
                vals = gates_1d[idx]
                out[cname] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(idx.sum())}
        return out

    @torch.no_grad()
    @ema_evaluation
    def evaluate(
        self,
        loader: DataLoader,
        use_ema: bool = False,
        return_arrays: bool = False,
        compute_auc: bool = True
    ) -> Dict[str, object]:
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        all_preds, all_targets = [], []
        all_probs = [] if compute_auc else None
        all_gates = []
        total_loss, correct, n = 0.0, 0, 0

        class_names = self.train_loader.dataset.classes
        class_to_idx = self.train_loader.dataset.class_to_idx

        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(self.cfg.device, non_blocking=True)
            y = y.to(self.cfg.device, non_blocking=True)

            with get_autocast_ctx(self.cfg.use_amp):
                out = self.model(x, return_gate=True, gate_alpha=1.0)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    logits, gate = out
                else:
                    logits, gate = out, None

                loss = F.cross_entropy(logits, y, weight=self.class_weights)

            probs = F.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)

            correct += (preds == y).sum().item()
            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

            if compute_auc and all_probs is not None:
                all_probs.append(probs.cpu().numpy())

            if gate is not None:
                all_gates.append(gate.detach().float().cpu().numpy())

        if use_ema and self.ema is not None:
            self.ema.restore()

        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        macro_f1 = f1_score(all_targets_np, all_preds_np, average="macro")
        micro_f1 = f1_score(all_targets_np, all_preds_np, average="micro")
        weighted_f1 = f1_score(all_targets_np, all_preds_np, average="weighted")
        macro_precision = precision_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
        macro_recall = recall_score(all_targets_np, all_preds_np, average="macro", zero_division=0)

        macro_auc = -1.0
        all_probs_np = None
        if compute_auc and all_probs is not None:
            all_probs_np = np.concatenate(all_probs, axis=0).astype(np.float64)
            try:
                macro_auc = roc_auc_score(
                    all_targets_np,
                    all_probs_np,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(len(class_names)),
                )
            except Exception as e:
                print(f"AUC computation failed: {e}")
                macro_auc = -1.0

        out_dict: Dict[str, object] = {
            "loss": float(total_loss / max(1, n)),
            "acc1": float(correct / max(1, n)),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_auc": float(macro_auc),
        }

        if return_arrays:
            out_dict["preds"] = all_preds_np
            out_dict["targets"] = all_targets_np
            if compute_auc and all_probs_np is not None:
                out_dict["probs"] = all_probs_np

        if self.cfg.use_hsv_branch and len(all_gates) > 0:
            gates_np = np.concatenate(all_gates, axis=0).reshape(-1)
            out_dict["gate_stats"] = self._compute_gate_stats(gates_np, all_targets_np, class_names, class_to_idx)
            out_dict["gate_global_mean"] = float(gates_np.mean())
            if return_arrays:
                out_dict["gates"] = gates_np

        return out_dict

    @torch.no_grad()
    def benchmark_inference(self, loader: DataLoader, warmup_batches: int = 5) -> Tuple[float, float]:
        self.model.eval()

        total_batches = len(loader)
        if total_batches <= warmup_batches:
            warnings.warn(
                f"Insufficient batches for benchmark: {total_batches} total, "
                f"{warmup_batches} warmup needed. Skipping throughput measurement."
            )
            return -1.0, -1.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = None
        n_images = 0

        for bi, (x, _y) in enumerate(loader):
            x = x.to(self.cfg.device, non_blocking=True)
            with get_autocast_ctx(self.cfg.use_amp):
                _ = self.model(x, gate_alpha=1.0)

            if bi == warmup_batches:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                n_images = 0

            if start_time is not None:
                n_images += x.size(0)

        if start_time is None or n_images == 0:
            warnings.warn("Benchmark failed: no images processed post-warmup")
            return -1.0, -1.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        throughput = n_images / max(elapsed, 1e-9)

        print(f"   Benchmark details: {n_images} images in {elapsed:.2f}s")
        return float(elapsed), float(throughput)

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, object], is_best: bool):
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        ckpt = {
            "epoch": int(epoch),
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg.__dict__,
            "best_val_f1": float(self.best_val_f1),
            "bad_epochs": int(self.bad_epochs),
            "train_log": self.train_log,
        }

        if self.ema is not None:
            ckpt["ema_shadow"] = {k: v.detach().clone().cpu() for k, v in self.ema.shadow.items()}

        ckpt.update(checkpoint_contract(self))

        self.ckpt_manager.save_checkpoint(
            checkpoint=ckpt,
            epoch=epoch,
            is_best=is_best,
            save_epoch_ckpt=self.cfg.save_epoch_checkpoints,
            epoch_interval=self.cfg.save_epoch_every,
            keep_n=self.cfg.keep_last_n_checkpoints
        )

    def save_train_log(self):
        if not self.train_log:
            return
        csv_path = self.logs_dir / "train_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.train_log[0].keys())
            writer.writeheader()
            writer.writerows(self.train_log)

    def fit(self):
        print("\n" + "=" * 80)
        print("START TRAINING")
        print("=" * 80)
        print(f"Experiment: {self.cfg.experiment_name}")
        print(f"Model:      {self.cfg.model_name}")
        print(f"Device:     {self.cfg.device}")
        print(f"AMP:        {self.cfg.use_amp and torch.cuda.is_available()}")
        print(f"EMA:        {self.cfg.use_ema}")
        print(f"GradAccum:  {self.cfg.gradient_accumulation_steps}")
        print("=" * 80)

        diag_path = self.ckpt_manager.final_dir / "metrics" / "init_scale_diagnostics.json"
        if (self.cfg.use_hsv_branch and self.cfg.log_init_scale_diagnostics
                and self.start_epoch == 1 and not diag_path.exists()):
            self._write_initialization_scale_diagnostics()

        effective_val_ema_used = False

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_m = self.train_one_epoch(epoch)

            use_ema_for_val = self.cfg.use_ema and (self.ema is not None)
            if self.cfg.use_ema and not use_ema_for_val:
                warnings.warn("EMA requested but not available - using raw weights for validation")

            val_m = self.evaluate(
                self.val_loader,
                use_ema=use_ema_for_val,
                return_arrays=False,
                compute_auc=self.cfg.compute_val_auc
            )

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            ema_tag = " (EMA)" if use_ema_for_val else ""
            print(f"\nEpoch {epoch}/{self.cfg.epochs} | LR: {current_lr:.2e}")
            print(f"  Train Loss: {train_m['loss']:.4f} | Train Acc@1: {train_m['acc1']:.4f}")
            print(
                f"  Val{ema_tag} Loss: {val_m['loss']:.4f} | "
                f"Acc@1: {val_m['acc1']:.4f} | Macro-F1: {val_m['macro_f1']:.4f} | "
                f"Micro-F1: {val_m['micro_f1']:.4f}"
            )

            if self.cfg.use_hsv_branch and "gate_stats" in val_m and "gate_global_mean" in val_m:
                self._append_gate_csv(epoch, val_m["gate_stats"], val_m["gate_global_mean"])

            self.train_log.append({
                "epoch": int(epoch),
                "lr": float(current_lr),
                "train_loss": float(train_m["loss"]),
                "train_acc1": float(train_m["acc1"]),
                "val_loss": float(val_m["loss"]),
                "val_acc1": float(val_m["acc1"]),
                "val_macro_f1": float(val_m["macro_f1"]),
                "val_micro_f1": float(val_m["micro_f1"]),
                "val_weighted_f1": float(val_m["weighted_f1"]),
                "val_macro_precision": float(val_m["macro_precision"]),
                "val_macro_recall": float(val_m["macro_recall"]),
                "val_macro_auc": float(val_m["macro_auc"]),
                "val_gate_global_mean": float(val_m.get("gate_global_mean", 0.0)),
            })
            self.save_train_log()

            improved = float(val_m["macro_f1"]) > self.best_val_f1
            if improved:
                effective_val_ema_used = use_ema_for_val
                self.best_val_f1 = float(val_m["macro_f1"])
                self.bad_epochs = 0
                self.save_checkpoint(epoch, val_m, is_best=True)
                print(f" New best Macro-F1: {self.best_val_f1:.4f}")
            else:
                self.bad_epochs += 1
                self.save_checkpoint(epoch, val_m, is_best=False)
                print(f"  No improvement ({self.bad_epochs}/{self.cfg.early_stopping_patience})")

            if self.bad_epochs >= self.cfg.early_stopping_patience:
                print("\n Early stopping triggered.")
                break

        print("\n" + "=" * 80)
        print("FINAL TEST EVALUATION")
        print("=" * 80)

        best_path = self.ckpt_manager.temp_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No best_model.pth found in {self.ckpt_manager.temp_dir}")

        best_ckpt = torch.load(best_path, map_location=self.cfg.device, weights_only=False)
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        if self.cfg.use_ema and "ema_shadow" in best_ckpt:
            print(f"Loading EMA weights from best checkpoint (epoch {best_ckpt.get('epoch', 'N/A')})")
            ema_state = dict(best_ckpt["model"])
            for k, v in best_ckpt["ema_shadow"].items():
                if k in ema_state:
                    ema_state[k] = v.to(self.cfg.device, non_blocking=True)
                else:
                    warnings.warn(f"EMA shadow key '{k}' not found in model state")
            load_state_dict_robust(target_model, ema_state)
            print("Loaded EMA weights for test evaluation")
            use_ema_for_test = False
            used_ema_weights = True
        elif self.cfg.use_ema:
            warnings.warn("EMA enabled but ema_shadow not found in checkpoint. Using raw weights.")
            load_state_dict_robust(target_model, best_ckpt["model"])
            use_ema_for_test = False
            used_ema_weights = False
        else:
            print(f"Loading raw weights from best checkpoint (epoch {best_ckpt.get('epoch', 'N/A')})")
            load_state_dict_robust(target_model, best_ckpt["model"])
            use_ema_for_test = False
            used_ema_weights = False

        test_m = self.evaluate(
            self.test_loader,
            use_ema=use_ema_for_test,
            return_arrays=True,
            compute_auc=True
        )
        test_m["used_ema_weights"] = used_ema_weights
        test_m.update(selection_metadata(self, best_ckpt, used_ema_weights))

        test_root = (Path(self.cfg.data_root) / "test").resolve()
        sample_ids = []
        for path, _label in self.test_loader.dataset.samples:
            pp = Path(path).resolve()
            try:
                sid = pp.relative_to(test_root).as_posix()
            except ValueError:
                sid = pp.name
            sample_ids.append(sid)
        if len(sample_ids) != len(test_m["targets"]):
            raise RuntimeError("test sample ID count does not match evaluated targets")
        test_m["sample_ids"] = sample_ids

        infer_s, throughput = self.benchmark_inference(self.test_loader, warmup_batches=5)
        test_m["inference_seconds"] = float(infer_s)
        test_m["throughput_img_s"] = float(throughput)
        test_m["params_m"] = float(self.params_m)
        test_m["gflops"] = float(self.gflops_out)

        test_m["model_selection"] = {
            "criterion": "macro_f1",
            "cfg_use_ema": bool(self.cfg.use_ema),
            "effective_use_ema_for_validation": bool(effective_val_ema_used),
            "effective_use_ema_for_test": bool(used_ema_weights),
            "ema_decay": float(self.cfg.ema_decay) if self.cfg.use_ema else None,
            "best_epoch": int(best_ckpt.get("epoch", -1)),
            "best_val_f1": float(self.best_val_f1),
            "used_class_weights": bool(self.cfg.use_class_weights),
        }

        print(f"\n Test Results:")
        print(f"   Acc@1:         {test_m['acc1']:.4f}")
        print(f"   Macro-F1:      {test_m['macro_f1']:.4f}")
        if test_m["macro_auc"] >= 0:
            print(f"   Macro-AUC:     {test_m['macro_auc']:.4f}")
        else:
            print(f"   Macro-AUC:     N/A")

        print(f"\n Model Complexity:")
        print(f"   Params:        {test_m['params_m']:.2f}M")
        if test_m["gflops"] >= 0:
            print(f"   GFLOPs:        {test_m['gflops']:.2f}")
        else:
            print(f"   GFLOPs:        N/A")

        if test_m["inference_seconds"] >= 0:
            print(f"\n Inference Speed (post-warmup, forward-only):")
            print(f"   Time:          {test_m['inference_seconds']:.2f}s")
            print(f"   Throughput:    {test_m['throughput_img_s']:.1f} img/s")

        self.ckpt_manager.copy_final_artifacts(
            best_ckpt_path=best_path,
            config=self.cfg,
            train_log=self.train_log,
            test_metrics=test_m,
            class_to_idx=self.train_loader.dataset.class_to_idx,
            classes=self.train_loader.dataset.classes,
            logs_dir=self.logs_dir,
        )

        return test_m


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train Swin (RGB baseline + optional RGB+HSV gated fusion) - stabilized HSV + gate warmup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--data_root", type=str, default="/kaggle/working/tea_leaf_clean")
    p.add_argument("--run_dir", type=str, default="/kaggle/working/experiments_swin")
    p.add_argument("--exp_name", type=str, required=True)

    p.add_argument("--model_name", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--drop_path_rate", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--campaign_id", default="tea_leaf_v2")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--use_class_weights", action="store_true", help="Use inverse frequency class weights")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--compute_val_auc", action="store_true")

    p.add_argument("--ckpt_temp_dir", type=str, default="/kaggle/temp")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--auto_resume", action="store_true")

    p.add_argument("--use_hsv", action="store_true")
    p.add_argument("--gate_vector", action="store_true")

    # HSV stabilization knobs
    p.add_argument("--hsv_raw", action="store_true", help="Use raw HSV (3ch) instead of sin/cos hue (4ch).")
    p.add_argument("--gate_warmup_epochs", type=int, default=5, help="Gate alpha warmup epochs (0->1).")
    p.add_argument("--hue_jitter", type=float, default=0.0,
                   help="Training hue jitter. Keep identical across R0-R3; paper recipe uses 0.0.")
    p.add_argument("--log_init_scale_diagnostics", action="store_true",
                   help="For HSV runs, record pre-training RGB/HSV RMS and gate contribution without changing training state.")

    p.add_argument("--save_epoch_checkpoints", action="store_true")
    p.add_argument("--save_epoch_every", type=int, default=5)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=3)

    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--no_cm_png", action="store_true")
    p.add_argument("--strict_final_counts", action="store_true",
                   help="Assert the frozen audited 7714-instance split counts before training.")

    return p.parse_args()


# -------------------------
# Optional cleaned-protocol sanity check
# -------------------------
def assert_clean_tealeafbd_counts(data_root: Union[str, Path]) -> None:
    """Assert the frozen audited 7,714-instance experiment partition."""
    root = Path(data_root)
    expected = {
        "train": {
            "Brown Blight": 858, "Gray Blight": 896,
            "Green mirid bug": 897, "Healthy leaf": 893,
            "Helopeltis": 863, "Red spider": 835,
            "Tea algal leaf spot": 848,
        },
        "val": {
            "Brown Blight": 83, "Gray Blight": 163,
            "Green mirid bug": 185, "Healthy leaf": 147,
            "Helopeltis": 86, "Red spider": 76,
            "Tea algal leaf spot": 76,
        },
        "test": {
            "Brown Blight": 86, "Gray Blight": 160,
            "Green mirid bug": 179, "Healthy leaf": 141,
            "Helopeltis": 89, "Red spider": 75,
            "Tea algal leaf spot": 78,
        },
    }
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        raise FileNotFoundError(f"Audited data_root not found: {root}")
    grand_total = 0
    for split, cls_counts in expected.items():
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        split_total = 0
        for cls, exp_n in cls_counts.items():
            cls_dir = split_root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")
            got_n = sum(1 for q in cls_dir.iterdir() if q.suffix.lower() in exts)
            if got_n != exp_n:
                raise AssertionError(
                    f"Unexpected count for {split}/{cls}: got {got_n}, expected {exp_n}"
                )
            split_total += got_n
        if split_total != sum(cls_counts.values()):
            raise AssertionError(f"Unexpected {split} total: {split_total}")
        grand_total += split_total
    if grand_total != 7714:
        raise AssertionError(f"Unexpected audited dataset total: {grand_total}, expected 7714")
    print("Audited TeaLeaf working-set count check passed: "
          "train=6090, val=816, test=808, total=7714")


def main():
    args = parse_args()

    if getattr(args, "strict_final_counts", False):
        assert_clean_tealeafbd_counts(args.data_root)

    cfg = Config(
        campaign_id=args.campaign_id,
        model_name=args.model_name,
        pretrained=args.pretrained,
        drop_path_rate=args.drop_path_rate,
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        use_ema=args.use_ema,
        use_class_weights=args.use_class_weights,
        experiment_name=args.exp_name,
        use_hsv_branch=args.use_hsv,
        gate_vector=args.gate_vector,
        hsv_use_sincos=(not args.hsv_raw),
        gate_warmup_epochs=int(args.gate_warmup_epochs),
        hue_jitter=float(args.hue_jitter),
        log_init_scale_diagnostics=bool(args.log_init_scale_diagnostics),
        ckpt_temp_dir=args.ckpt_temp_dir,
        save_epoch_checkpoints=args.save_epoch_checkpoints,
        save_epoch_every=args.save_epoch_every,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compute_val_auc=args.compute_val_auc,
        run_dir=args.run_dir,
        use_data_parallel=(not args.no_dp),
        save_cm_png=(not args.no_cm_png),
    )

    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Num GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Timm:    {timm.__version__}")
    print("=" * 80)

    trainer = Trainer(cfg)

    if args.resume is not None:
        trainer.resume_from(Path(args.resume))
    elif args.auto_resume:
        last_ckpt = trainer.ckpt_manager.temp_dir / "last_checkpoint.pth"
        if last_ckpt.exists():
            trainer.resume_from(last_ckpt)
        else:
            print(" auto_resume enabled but no last_checkpoint.pth found. Starting fresh.")

    trainer.fit()


if __name__ == "__main__":
    main()