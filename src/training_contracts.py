"""Shared v2 training, checkpoint and dataset contracts."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import json
import platform
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

PROTOCOL = "tea_leaf_v2"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint requires CUDA for exact resume")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def rng_neutral(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        state = rng_state()
        try:
            return fn(*args, **kwargs)
        finally:
            restore_rng(state)
    return wrapped


def matched_initialization(fn):
    """Keep construction RNG-neutral and initialize the common head from seed+1."""
    @wraps(fn)
    @rng_neutral
    def wrapped(self, *args, **kwargs):
        seed = torch.initial_seed()
        fn(self, *args, **kwargs)
        torch.manual_seed(seed + 1)
        for layer in self.classifier.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
    return wrapped


class EpochDataset(Dataset):
    """Per-image/epoch augmentation RNG independent of architecture and worker count."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0

    def __getattr__(self, key):
        if key == "dataset":
            raise AttributeError(key)
        return getattr(self.dataset, key)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seed = int.from_bytes(hashlib.sha256(
            f"{self.seed}:{self.epoch}:{index}".encode()).digest()[:4], "little")
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)


def start_epoch(loader, epoch, seed):
    if loader.persistent_workers:
        raise RuntimeError("EpochDataset requires non-persistent workers")
    loader.dataset.epoch = epoch
    loader.generator.manual_seed(seed + epoch)


def optimizer_groups(model, decay):
    regular, exempt = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (exempt if param.ndim <= 1 or "gate" in name else regular).append(param)
    return [{"params": regular, "weight_decay": decay},
            {"params": exempt, "weight_decay": 0.0}]


def accumulation_weight(loader, batch_index, accumulation, batch_size):
    """Sample-weighted mean over the actual update group, including its tail."""
    first = ((batch_index - 1) // accumulation) * accumulation
    samples = min(accumulation * loader.batch_size,
                  len(loader.dataset) - first * loader.batch_size)
    return batch_size / samples


def optimizer_update(trainer):
    before = trainer.scaler.get_scale()
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    updated = trainer.scaler.get_scale() >= before
    if updated:
        trainer.scheduler.step()
        if trainer.ema is not None:
            trainer.ema.update()
        trainer.optimizer_updates = getattr(trainer, "optimizer_updates", 0) + 1
    else:
        trainer.skipped_updates = getattr(trainer, "skipped_updates", 0) + 1
    return updated


def ema_evaluation(fn):
    @wraps(fn)
    def wrapped(self, loader, use_ema=False, *args, **kwargs):
        training = self.model.training
        active = use_ema and self.ema is not None
        if active:
            self.ema.apply_shadow()
        try:
            return fn(self, loader, *args, use_ema=False, **kwargs)
        finally:
            if active:
                self.ema.restore()
            self.model.train(training)
    return wrapped


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def immutable_config(directory, cfg):
    directory = Path(directory)
    path = directory / "config.json"
    value = json.loads(json.dumps(cfg.__dict__))
    # Metrics are exported before the remaining artifacts. Only the final marker
    # commits a completed run; an interrupted export must remain resumable.
    if (directory.parent / "COMPLETE.json").exists():
        raise RuntimeError(f"Completed run is immutable: {directory.parent}")
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("Run configuration changed; use a new run directory")
    else:
        path.write_text(json.dumps(value, indent=2))
    env = directory / "env.json"
    if not env.exists():
        import timm
        env.write_text(json.dumps({"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "timm": timm.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))


def checkpoint_contract(trainer):
    return {"protocol": PROTOCOL, "rng_state": rng_state(),
            "loader_rng": trainer.train_loader.generator.get_state(),
            "source_hashes": source_hashes(),
            "optimizer_updates": getattr(trainer, "optimizer_updates", 0),
            "skipped_updates": getattr(trainer, "skipped_updates", 0)}


def resume_training(trainer, path, model, strict_load):
    # Only load checkpoints produced by this repository from trusted local paths.
    ckpt = torch.load(path, map_location=trainer.cfg.device, weights_only=False)
    if ckpt.get("protocol") != PROTOCOL or ckpt.get("source_hashes") != source_hashes():
        raise RuntimeError("Checkpoint protocol/source differs; start a new campaign")
    if json.loads(json.dumps(ckpt["config"])) != json.loads(json.dumps(trainer.cfg.__dict__)):
        raise RuntimeError("Checkpoint configuration differs (including epoch budget)")
    strict_load(model, ckpt["model"])
    for name in ("optimizer", "scheduler", "scaler"):
        getattr(trainer, name).load_state_dict(ckpt[name])
    if trainer.ema is not None:
        shadow = ckpt["ema_shadow"]
        if set(shadow) != set(trainer.ema.shadow):
            raise RuntimeError("EMA checkpoint parameter keys differ")
        trainer.ema.shadow = {k: v.to(trainer.cfg.device) for k, v in shadow.items()}
    trainer.best_val_f1 = float(ckpt["best_val_f1"])
    trainer.bad_epochs = int(ckpt["bad_epochs"])
    trainer.start_epoch = int(ckpt["epoch"]) + 1
    trainer.train_log = ckpt["train_log"]
    trainer.optimizer_updates = ckpt["optimizer_updates"]
    trainer.skipped_updates = ckpt["skipped_updates"]
    trainer.train_loader.generator.set_state(ckpt["loader_rng"].cpu())
    restore_rng(ckpt["rng_state"])
    history = Path(trainer.config_dir) / "resume_history.jsonl"
    with history.open("a") as f:
        f.write(json.dumps({"checkpoint": str(Path(path).resolve()),
                            "resumed_after_epoch": ckpt["epoch"]}) + "\n")


def selection_metadata(trainer, checkpoint, used_ema):
    required = {"epoch", "best_val_f1", "optimizer_updates", "skipped_updates"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"Selected checkpoint missing metadata: {sorted(missing)}")
    return {"campaign_id": trainer.cfg.campaign_id, "protocol": PROTOCOL,
            "best_epoch": int(checkpoint["epoch"]),
            "best_val_f1": float(checkpoint["best_val_f1"]),
            "selection_criterion": "validation_macro_f1",
            "validation_uses_ema": bool(trainer.cfg.use_ema),
            "used_ema_weights": bool(used_ema), "ema_decay": trainer.cfg.ema_decay,
            # These counters describe the selected checkpoint, not the final epoch.
            "optimizer_updates": int(checkpoint["optimizer_updates"]),
            "skipped_updates": int(checkpoint["skipped_updates"]),
            "final_optimizer_updates": int(getattr(trainer, "optimizer_updates", 0)),
            "final_skipped_updates": int(getattr(trainer, "skipped_updates", 0)),
            "best_at_budget_boundary": int(checkpoint["epoch"]) == trainer.cfg.epochs,
            "eval_loss_definition": "unsmoothed_cross_entropy"}


def validate_eval_index(dataset, split, data_root):
    path = Path(data_root).parent / "manifests/final_decontam_index.json"
    index = json.loads(path.read_text())
    expected = index[split]
    order = [Path(p).name for p, _ in dataset.samples]
    labels = [int(y) for _, y in dataset.samples]
    digest = hashlib.sha256("\n".join(order).encode()).hexdigest()
    if (list(dataset.classes) != index["classes"] or order != expected["order"]
            or labels != expected["labels"] or digest != expected["sha256"]):
        raise RuntimeError(f"{split} loader disagrees with frozen evaluation index")
