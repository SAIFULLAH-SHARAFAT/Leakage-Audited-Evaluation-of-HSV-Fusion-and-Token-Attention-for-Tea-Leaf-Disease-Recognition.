
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import io
import traceback
import time

# The served model is the best-performing run in the campaign
# (results/v2/runs/C1_s1337), which is arch="rgb".
from model import SwinRGB
from logic import (
    generate_explanations,
    generate_consensus_heatmap,
    compute_agreement,
    run_quality_checks,
    AGREEMENT_IOT_THRESHOLD,
    ENERGY_RATIO_THRESHOLD,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = [
    "Brown Blight", "Gray Blight", "Green mirid bug", "Healthy leaf",
    "Helopeltis", "Red spider", "Tea algal leaf spot",
]

# ── Image quality gate thresholds ─────────────────────────────────────────────
BLUR_THRESHOLD   = 4
DARK_THRESHOLD   = 30
BRIGHT_THRESHOLD = 220

# ── Confidence threshold ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70

# ── Checkpoint served by this API ─────────────────────────────────────────────
# Best-performing campaign run: results/v2/runs/C1_s1337 (arch="rgb").
# If you later want to serve the TLA (tcca) run instead, see the comment
# at the bottom of get_model() — it is a 3-line switch.
CHECKPOINT_PATH = "/SLURM/home/slurm_g202621260/tealeaf/results/v2/runs/C1_s1337/model/ema_model_weights_only.pth"

def check_image_quality(img_np: np.ndarray) -> dict:
    gray       = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_pixel = float(gray.mean())

    is_blurry = blur_score < BLUR_THRESHOLD
    is_dark   = mean_pixel < DARK_THRESHOLD
    is_bright = mean_pixel > BRIGHT_THRESHOLD

    return {
        "passed":     not (is_blurry or is_dark or is_bright),
        "blur_score": round(blur_score, 2),
        "mean_pixel": round(mean_pixel, 2),
        "is_blurry":  is_blurry,
        "is_dark":    is_dark,
        "is_bright":  is_bright,
        "thresholds": {
            "blur":   BLUR_THRESHOLD,
            "dark":   DARK_THRESHOLD,
            "bright": BRIGHT_THRESHOLD,
        },
    }


def get_model():
    print(f"Loading model weights from {CHECKPOINT_PATH}...")
    model = SwinRGB(num_classes=len(CLASSES), pretrained=False)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Unpack EMA shadow / state_dict / raw dict
    if isinstance(checkpoint, dict):
        if "ema_shadow" in checkpoint:
            sd = checkpoint["ema_shadow"]
            print("Loading EMA shadow weights")
        elif "model" in checkpoint:
            sd = checkpoint["model"]
            print("Loading model state_dict from checkpoint")
        else:
            sd = checkpoint
            print("Loading raw checkpoint dict")
    else:
        sd = checkpoint
        print("Loading raw weights")

    # Strip DataParallel "module." prefix if present
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        print("Stripped 'module.' prefix from checkpoint keys")

    # Guard: this server serves arch="rgb". Refuse other-arch checkpoints.
    token_prefixes = ("tcca.", "color_proj4.", "color_encoder.", "adapter.", "eca.")
    offenders = [k for k in sd if k.startswith(token_prefixes)]
    if offenders:
        raise RuntimeError(
            f"Checkpoint contains token-module keys (e.g. {offenders[:3]}); "
            "it was not saved from arch='rgb'. Swap SwinRGB for SwinTCCA "
            "in the import at the top of main.py to serve a TLA run."
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}):")
        for k in missing:
            print(f"   {k}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"   {k}")
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint / model state_dict mismatch — refusing to serve a "
            "partially-initialized model."
        )

    print(f"All {len(sd)} keys matched — arch='rgb' model loaded successfully")
    model.to(device)
    model.eval()
    return model


model = get_model()


def find_last_swin_norm1(model):
    """Locate Swin stage-4 block-(-1) norm1 for Grad-CAM, robust to wrapping.

    timm's Swin stores stages as `self.layers = nn.Sequential(*layers)`, so
    we cannot rely on ModuleList — accept any container that supports len()
    and indexing, and verify the last stage exposes `.blocks`.
    """
    if hasattr(model, "module"):            # unwrap DataParallel/DDP
        model = model.module

    for _, m in model.named_modules():
        layers = getattr(m, "layers", None)
        if layers is None:
            continue
        try:
            n = len(layers)
        except TypeError:
            continue
        if n != 4:
            continue
        last = layers[-1]
        blocks = getattr(last, "blocks", None)
        if blocks is None or len(blocks) == 0:
            continue
        return blocks[-1].norm1

    raise RuntimeError("Could not locate Swin stage ModuleList for Grad-CAM")


target_layer = find_last_swin_norm1(model)
print(f"Target layer: {target_layer}")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents    = await file.read()
        image_pil   = Image.open(io.BytesIO(contents)).convert("RGB")
        img_uint8   = np.array(image_pil)
        original_np = img_uint8 / 255.0

        quality_gate = check_image_quality(img_uint8)
        if not quality_gate["passed"]:
            reason = (
                "Image is too blurry" if quality_gate["is_blurry"] else
                "Image is underexposed (too dark)" if quality_gate["is_dark"] else
                "Image is overexposed (too bright)"
            )
            return {
                "final_status":  "image_rejected",
                "reject_reason": reason,
                "image_quality": quality_gate,
                "prediction":    None,
                "agreement_score": None,
                "quality_flags": None,
                "xai_stats":     None,
                "explanations":  None,
                "inference_ms":  0,
            }

        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs  = torch.nn.functional.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        predicted_label = CLASSES[pred_idx.item()]
        conf_score      = float(conf.item())
        is_uncertain    = conf_score < CONFIDENCE_THRESHOLD

        explanations, masks = {}, {}
        agreement_score     = 0.0
        quality_flags       = {"background": False, "spread": False, "border": False}
        xai_stats           = None

        if not is_uncertain:
            with torch.set_grad_enabled(True):
                explanations, masks = generate_explanations(
                    model, input_tensor, target_layer, original_np
                )
            merged = generate_consensus_heatmap(masks, original_np)
            explanations["consensus"] = merged if merged else explanations.get("gradcam", "")

            agreement_score = compute_agreement(masks)
            quality_flags   = run_quality_checks(masks, agreement_score)
            agreement_pass  = quality_flags.pop("agreement_pass")
            energy_ratios   = quality_flags.pop("energy_ratios")
            mean_energy     = float(np.mean(list(energy_ratios.values()))) if energy_ratios else 0.0

            xai_stats = {
                "agreement_pass":    agreement_pass,
                "mean_energy_ratio": round(mean_energy, 3),
                "energy_pass":       mean_energy >= ENERGY_RATIO_THRESHOLD,
                "per_method_energy": energy_ratios,
                "thresholds": {
                    "agreement_iou": AGREEMENT_IOT_THRESHOLD,
                    "energy_ratio":  ENERGY_RATIO_THRESHOLD,
                },
            }

            any_issue    = any(quality_flags.values())
            final_status = "retake_required" if (not agreement_pass or any_issue) else "accepted"
        else:
            final_status = "uncertain"

        elapsed = round((time.time() - start_time) * 1000, 1)

        return {
            "prediction":      {"label": predicted_label, "confidence": round(conf_score, 3)},
            "is_uncertain":    is_uncertain,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "agreement_score": round(agreement_score, 3),
            "quality_flags":   quality_flags,
            "final_status":    final_status,
            "image_quality":   quality_gate,
            "xai_stats":       xai_stats,
            "explanations":    explanations,
            "inference_ms":    elapsed,
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "architecture": "rgb",             # served run: C1_s1337
        "checkpoint":   CHECKPOINT_PATH,
        "classes":      len(CLASSES),
        "device":       str(device),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
