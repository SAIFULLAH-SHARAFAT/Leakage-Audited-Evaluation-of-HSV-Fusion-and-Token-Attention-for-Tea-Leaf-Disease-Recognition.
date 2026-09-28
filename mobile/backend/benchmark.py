import torch
import torch.nn as nn
import time
import os
import tempfile
import timm

CHECKPOINT  = "/SLURM/home/slurm_g202621260/tealeaf/results/v2/runs/C1_s1337/model/ema_model_weights_only.pth"
NUM_CLASSES = 7
WARMUP      = 50
ITERATIONS  = 100


# ── C1 model: swin + classifier, no token module ─────────────────────────────
class SwinWithHooks(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model(
            "swin_small_patch4_window7_224.ms_in1k",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.2,
            global_pool="",
        )

    def forward(self, x):
        tokens = self.swin(x)           # (B, 49, 768) after final LayerNorm
        return tokens.mean(dim=1)       # (B, 768)


class C1Model(nn.Module):
    """
    Backbone + classifier only — matches arch='rgb' in train_token_models.py.
    Keys: img_mean, img_std, swin.swin.*, classifier.*
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.register_buffer("img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.swin = SwinWithHooks()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.swin(x))


def benchmark_model():
    print("-" * 50)
    print("Starting Model Benchmark (C1 — backbone + classifier)")
    print("-" * 50)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device}")

    # Load
    model = C1Model(num_classes=NUM_CLASSES)
    ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = ck.get("ema_shadow", ck.get("model", ck)) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Missing keys:    {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    model.to(device).eval()

    # Params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:  {total_params:,}  ({total_params/1e6:.2f} M)")

    # State-dict size
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        os.unlink(tmp.name)
    print(f"State-dict size:   {size_mb:.1f} MB")

    def sync():
        if device.type == "cuda":   torch.cuda.synchronize()
        elif device.type == "mps":  torch.mps.synchronize()

    # ── Model-only inference ──────────────────────────────────────────────
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(WARMUP): model(dummy); sync()
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(ITERATIONS): model(dummy); sync()
    inference_ms = (time.perf_counter() - t0) / ITERATIONS * 1000

    # ── Full pipeline ─────────────────────────────────────────────────────
    import io, numpy as np
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ).save(tmp_img.name, quality=85)
    tmp_img.close()

    times = {k: [] for k in ["disk_io","preprocess","transfer","inference","postprocess"]}

    with torch.no_grad():
        for _ in range(WARMUP):
            img = Image.open(tmp_img.name).convert("RGB")
            t = transform(img).unsqueeze(0).to(device); sync()
            model(t); sync()

        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            raw = open(tmp_img.name, "rb").read()
            t1 = time.perf_counter()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            t2 = time.perf_counter()
            tensor = tensor.to(device); sync()
            t3 = time.perf_counter()
            out = model(tensor); sync()
            t4 = time.perf_counter()
            probs = torch.nn.functional.softmax(out, dim=1)
            pred_idx = int(probs.reshape(-1).argmax().item())
            t5 = time.perf_counter()

            times["disk_io"].append((t1-t0)*1000)
            times["preprocess"].append((t2-t1)*1000)
            times["transfer"].append((t3-t2)*1000)
            times["inference"].append((t4-t3)*1000)
            times["postprocess"].append((t5-t4)*1000)

    os.unlink(tmp_img.name)

    def mean(lst): return sum(lst)/len(lst)

    disk    = mean(times["disk_io"])
    pre     = mean(times["preprocess"])
    xfer    = mean(times["transfer"])
    infer   = mean(times["inference"])
    post    = mean(times["postprocess"])
    total   = disk + pre + xfer + infer + post
    e2e     = total + 95.0

    print("\n" + "=" * 50)
    print("RESULTS — paste into Table XVI (\\pending{C1})")
    print("=" * 50)
    print(f"Parameters:          {total_params/1e6:.2f} M")
    print(f"State-dict size:     {size_mb:.1f} MB")
    print(f"GFLOPs:              17.12  (from ptflops)")
    print("-" * 50)
    print(f"Disk I/O:            {disk:.2f}")
    print(f"Preprocessing:       {pre:.2f}")
    print(f"CPU/GPU transfer:    {xfer:.2f}")
    print(f"Model inference:     {infer:.2f}")
    print(f"Post-processing:     {post:.2f}")
    print(f"Total server:        {total:.2f}")
    print(f"Assumed 4G latency:  95.00")
    print(f"End-to-end total:    {e2e:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    benchmark_model()
