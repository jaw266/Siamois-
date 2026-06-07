import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from scipy.ndimage import label, binary_fill_holes


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path, img_size, mean, std):
    image = Image.open(image_path).convert("RGB")
    original = image.copy()
    image = image.resize((img_size, img_size))

    arr = np.asarray(image).astype("float32") / 255.0
    mean_arr = np.array(mean, dtype="float32").reshape(1, 1, 3)
    std_arr  = np.array(std,  dtype="float32").reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr

    # ONNX expects [batch, channels, height, width]
    arr = np.transpose(arr, (2, 0, 1))[None, :, :, :].astype("float32")
    return arr, original


def augment_pair(a, b):
    """Retourne 4 paires augmentées (original, hflip, vflip, rot180) pour TTA."""
    def hflip(x): return x[:, :, :, ::-1].copy()
    def vflip(x): return x[:, :, ::-1, :].copy()
    return [
        (a, b),
        (hflip(a), hflip(b)),
        (vflip(a), vflip(b)),
        (hflip(vflip(a)), hflip(vflip(b))),
    ]


def deaugment_seg(seg, idx):
    """Inverse l'augmentation sur le masque de segmentation."""
    def hflip(x): return x[:, :, :, ::-1].copy()
    def vflip(x): return x[:, :, ::-1, :].copy()
    if idx == 0: return seg
    if idx == 1: return hflip(seg)
    if idx == 2: return vflip(seg)
    if idx == 3: return hflip(vflip(seg))


def postprocess_mask(mask, min_area, fill_holes):
    """Supprime les petites régions bruitées et remplit les trous."""
    cleaned = mask.copy()
    if fill_holes:
        cleaned = binary_fill_holes(cleaned).astype(np.uint8)
    if min_area > 0:
        labeled, num = label(cleaned)
        for region_id in range(1, num + 1):
            if (labeled == region_id).sum() < min_area:
                cleaned[labeled == region_id] = 0
    return cleaned


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
    }


def severity_level(cls_prob, area_ratio, pred_label, cls_thr):
    if pred_label == 0 and area_ratio < 0.001:
        return "NORMAL"
    if cls_prob >= 0.90 or area_ratio >= 0.03:
        return "HIGH"
    if cls_prob >= cls_thr or area_ratio >= 0.005:
        return "MEDIUM"
    return "LOW"


def save_mask_and_overlay(mask, image_t2_original, out_mask_path, out_overlay_path, alpha=0.45):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image_t2_original.size)
    mask_np = np.asarray(mask_img) > 0

    base = image_t2_original.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    overlay_np = np.asarray(overlay).copy()
    overlay_np[mask_np] = [255, 0, 0, int(255 * alpha)]
    overlay = Image.fromarray(overlay_np, mode="RGBA")

    result = Image.alpha_composite(base, overlay)
    mask_img.save(out_mask_path)
    result.convert("RGB").save(out_overlay_path)


def optimize_model(model_path):
    """Crée une version optimisée du modèle ONNX (graph optimizations)."""
    opt_path = str(model_path).replace(".onnx", "_optimized.onnx")
    if Path(opt_path).exists():
        return opt_path
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = opt_path
    ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"Modèle optimisé sauvegardé : {opt_path}")
    return opt_path


def run_inference(model_path, config_path, t1_path, t2_path, id_test, output_dir):
    cfg = load_config(config_path)
    img_size   = int(cfg["img_size"])
    cls_thr    = float(cfg["classification_threshold"])
    seg_thr    = float(cfg["segmentation_threshold"])
    min_area   = int(cfg.get("postprocess_min_area", 100))
    fill_holes = bool(cfg.get("postprocess_fill_holes", True))
    use_tta    = bool(cfg.get("use_tta", True))
    mean = cfg["mean"]
    std  = cfg["std"]

    image_A, _           = preprocess_image(t1_path, img_size, mean, std)
    image_B, t2_original = preprocess_image(t2_path, img_size, mean, std)

    # Utilise le modèle optimisé si disponible
    opt_path = optimize_model(model_path)
    active_model = opt_path if Path(opt_path).exists() else str(model_path)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(active_model, sess_options=opts, providers=["CPUExecutionProvider"])

    if use_tta:
        # TTA : 4 augmentations → moyenne des probabilités
        pairs = augment_pair(image_A, image_B)
        seg_sum = None
        cls_sum = 0.0
        for i, (a, b) in enumerate(pairs):
            out = session.run(None, {"image_A": a, "image_B": b})
            seg_aug = deaugment_seg(np.asarray(out[0]), i)
            seg_sum  = seg_aug if seg_sum is None else seg_sum + seg_aug
            cls_sum += float(np.asarray(out[1]).reshape(-1)[0])
        seg_prob = seg_sum / len(pairs)
        cls_prob = cls_sum / len(pairs)
    else:
        out = session.run(None, {"image_A": image_A, "image_B": image_B})
        seg_prob = np.asarray(out[0])
        cls_prob = float(np.asarray(out[1]).reshape(-1)[0])

    seg_2d    = np.squeeze(seg_prob)
    pred_mask = (seg_2d >= seg_thr).astype(np.uint8)
    pred_mask = postprocess_mask(pred_mask, min_area, fill_holes)
    area_ratio = float(pred_mask.mean())
    pred_label = int(cls_prob >= cls_thr)
    decision   = "CHANGE" if pred_label == 1 else "NO_CHANGE"
    severity   = severity_level(cls_prob, area_ratio, pred_label, cls_thr)
    bbox       = mask_bbox(pred_mask)

    out_dir      = Path(output_dir)
    masks_dir    = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    logs_dir     = out_dir / "logs"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mask_path    = masks_dir    / f"{id_test}_pred_mask.png"
    overlay_path = overlays_dir / f"{id_test}_overlay.png"
    save_mask_and_overlay(pred_mask, t2_original, mask_path, overlay_path)

    result = {
        "id_test": id_test,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_t1": str(t1_path),
        "image_t2": str(t2_path),
        "classification_probability": round(cls_prob, 6),
        "classification_threshold": cls_thr,
        "prediction": decision,
        "segmentation_threshold": seg_thr,
        "change_area_ratio": round(area_ratio, 6),
        "severity": severity,
        "bbox": bbox,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "tta_enabled": use_tta,
    }

    result_path = logs_dir / f"{id_test}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for bitemporal change detection")
    parser.add_argument("--model",      default="../model/siamese_change_detector_phase2bis.onnx")
    parser.add_argument("--config",     default="../model/deployment_config_phase2bis.json")
    parser.add_argument("--t1",         required=True)
    parser.add_argument("--t2",         required=True)
    parser.add_argument("--id_test",    default="test_01")
    parser.add_argument("--output_dir", default="../outputs")
    args = parser.parse_args()

    result = run_inference(args.model, args.config, args.t1, args.t2, args.id_test, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
