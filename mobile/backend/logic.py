
import cv2
import numpy as np
import torch
import base64
import itertools
from scipy import ndimage

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit

# ── Thresholds (mirror paper values) ─────────────────────────────────────────
AGREEMENT_IOT_THRESHOLD  = 0.35   # IoU ≥ 0.35  → inter-method agreement pass
ENERGY_RATIO_THRESHOLD   = 0.25   # energy ratio ≥ 0.25 → quality pass (recalibrated for macro leaf images)
TOP_K_FRACTION           = 0.10   # top-10 % of pixels used for binary masks

def tensor_to_cv2(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img

def encode_image(cv2_img):
    _, buffer = cv2.imencode(".png", cv2_img * 255)
    return base64.b64encode(buffer).decode("utf-8")

def normalize_heatmap(heatmap):
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - h_min) / (h_max - h_min)

def swin_reshape_transform(tensor):
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    height = width = int(np.sqrt(tensor.shape[1]))
    result = tensor.transpose(1, 2)
    result = result.reshape(tensor.shape[0], tensor.shape[2], height, width)
    return result


# ── HiResCAM (element-wise gradient × activation) ────────────────────────────
class HiResCAM:
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.activations = self.gradients = None

    def __call__(self, input_tensor, targets=None):
        # `targets` is accepted for call-site parity with the pytorch_grad_cam
        # extractors.  HiResCAM is inherently class-specific via the argmax
        # below, so we ignore the argument and use the same predicted class.
        self.model.eval()
        self.activations = self.gradients = None

        def forward_hook(module, input, output):
            self.activations = (
                self.reshape_transform(output) if self.reshape_transform else output
            )

        def backward_hook(module, grad_input, grad_output):
            self.gradients = (
                self.reshape_transform(grad_output[0])
                if self.reshape_transform
                else grad_output[0]
            )

        h_fwd = self.target_layers[0].register_forward_hook(forward_hook)
        h_bwd = self.target_layers[0].register_full_backward_hook(backward_hook)
        try:
            output = self.model(input_tensor)
            self.model.zero_grad()
            pred_idx = int(output.argmax(dim=1).item())
            output[0, pred_idx].backward(retain_graph=True)

            acts = self.activations.detach().cpu().numpy()
            grads = self.gradients.detach().cpu().numpy()
            cam = np.maximum((acts * grads).sum(axis=1), 0)
            return cam[0] if len(cam.shape) == 3 else cam
        finally:
            h_fwd.remove()
            h_bwd.remove()

    def release(self):
        pass


# ── CAM generation ────────────────────────────────────────────────────────────
def generate_explanations(model, input_tensor, target_layer, original_img):
    print("Starting XAI generation...")
    results, masks = {}, {}

    cams = {
        "gradcam":    GradCAM(model=model, target_layers=[target_layer],
                              reshape_transform=swin_reshape_transform),
        "gradcampp":  GradCAMPlusPlus(model=model, target_layers=[target_layer],
                                      reshape_transform=swin_reshape_transform),
        "layercam":   LayerCAM(model=model, target_layers=[target_layer],
                               reshape_transform=swin_reshape_transform),
        "ablationcam": AblationCAM(model=model, target_layers=[target_layer],
                                   reshape_transform=swin_reshape_transform,
                                   ablation_layer=AblationLayerVit()),
        "hirescam":   HiResCAM(model=model, target_layers=[target_layer],
                               reshape_transform=swin_reshape_transform),
    }

    orig_h, orig_w, _ = original_img.shape

    # ── Compute predicted class once; every CAM is class-specific on it ──────
    # pytorch_grad_cam's BaseCAM.__call__ requires `targets` explicitly in this
    # version — it has no default.  Passing None would default to argmax inside
    # the library, but we pin the class explicitly so the heatmap and the
    # predicted label returned by the API always agree.
    model.eval()
    with torch.no_grad():
        logits   = model(input_tensor)
        pred_idx = int(logits.argmax(dim=1).item())
    cam_targets = [ClassifierOutputTarget(pred_idx)]

    try:
        for name, cam_extractor in cams.items():
            try:
                print(f"Generating {name.upper()}...")
                if name == "ablationcam":
                    cam_extractor.batch_size = 2

                if name == "hirescam":
                    grayscale_cam = cam_extractor(input_tensor=input_tensor)
                else:
                    grayscale_cam = cam_extractor(
                        input_tensor=input_tensor,
                        targets=cam_targets,
                    )[0, :]

                norm_heatmap  = normalize_heatmap(grayscale_cam)
                resized       = cv2.resize(norm_heatmap, (orig_w, orig_h))
                masks[name]   = resized
                vis           = show_cam_on_image(original_img, resized, use_rgb=True)
                results[name] = encode_image(vis)

            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = ""
                masks[name]   = np.zeros((orig_h, orig_w))
    finally:
        for cam_extractor in cams.values():
            if hasattr(cam_extractor, "activations_and_grads"):
                cam_extractor.activations_and_grads.release()
            else:
                cam_extractor.release()

    print("XAI generation complete.")
    return results, masks


# ── Consensus heatmap ─────────────────────────────────────────────────────────
def generate_consensus_heatmap(masks, original_img):
    """Average all valid per-method normalised heatmaps → consensus overlay."""
    print("Generating consensus heatmap...")
    valid = [m for m in masks.values() if m.max() > 0]
    if not valid:
        return ""
    consensus = normalize_heatmap(np.mean(np.array(valid), axis=0))
    vis = show_cam_on_image(original_img, consensus, use_rgb=True)
    return encode_image(vis)


# ── Quantitative agreement score (mean pairwise IoU of top-k binary masks) ───
def compute_agreement(masks, k: float = TOP_K_FRACTION) -> float:
    """
    Return mean pairwise IoU over the top-k% active pixels of each valid mask.
    Mirrors the paper's inter-method agreement check (IoU ≥ AGREEMENT_IOT_THRESHOLD).
    """
    valid = {n: m for n, m in masks.items() if m.max() > 0}
    if not valid:
        return 0.0

    binary_masks = []
    for mask in valid.values():
        threshold = np.percentile(mask, 100 - k * 100)
        binary_masks.append((mask >= threshold).astype(float))

    ious = []
    for m1, m2 in itertools.combinations(binary_masks, 2):
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        ious.append(inter / union if union > 0 else 0.0)

    return float(np.mean(ious)) if ious else 0.0


# ── Energy-ratio quality check ────────────────────────────────────────────────
def compute_energy_ratio(mask: np.ndarray, k: float = TOP_K_FRACTION) -> float:
    """
    Fraction of the top-k active pixels that fall in the centre 60 % × 60 %
    of the image.  Values ≥ ENERGY_RATIO_THRESHOLD indicate the model is
    attending to the leaf rather than the background border.
    """
    if mask.max() == 0:
        return 0.0
    h, w = mask.shape
    h0, h1 = int(h * 0.2), int(h * 0.8)
    w0, w1 = int(w * 0.2), int(w * 0.8)
    threshold    = np.percentile(mask, 100 - k * 100)
    binary       = (mask >= threshold).astype(float)
    total        = binary.sum()
    if total == 0:
        return 0.0
    return float(binary[h0:h1, w0:w1].sum() / total)


def run_quality_checks(masks, agreement_score, k: float = TOP_K_FRACTION) -> dict:
    """
    Returns per-flag quality dict **and** the per-method energy ratios
    (used for the quantitative summary returned to the frontend).

    background flag: True when ≥ 2 methods fail the energy-ratio threshold,
                     meaning attention is concentrated outside the leaf centre.
    """
    valid = {n: m for n, m in masks.items() if m.max() > 0}
    if not valid:
        return {
            "background": False,
            "spread": False,
            "border": False,
            "energy_ratios": {},
            "agreement_pass": False,
        }

    energy_ratios = {name: compute_energy_ratio(mask, k) for name, mask in valid.items()}
    center_fails  = sum(1 for er in energy_ratios.values() if er < ENERGY_RATIO_THRESHOLD)

    return {
        "background":    center_fails >= 2,
        "spread":        False,
        "border":        False,
        # ── quantitative fields for frontend display & paper stats ──
        "energy_ratios": {n: round(v, 3) for n, v in energy_ratios.items()},
        "agreement_pass": agreement_score >= AGREEMENT_IOT_THRESHOLD,
    }
