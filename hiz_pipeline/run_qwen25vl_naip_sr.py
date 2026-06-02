"""
run_qwen25vl_naip_sr.py
HIZ-VLM Pipeline — Qwen2.5-VL Inference on SR-NAIP Imagery

Runs Qwen2.5-VL-7B on Real-ESRGAN 4× super-resolved NAIP tiles (~15 cm GSD).
Identical pipeline to run_qwen25vl.py (MLX backend, Graph-RAG prompts, Geo-CoT)
but on AI-upscaled public imagery.

Results stored in ~/hiz_pipeline/results/naip_sr_qwen25vl/

Scientific purpose: ablation — does AI super-resolution of 60 cm public
imagery close the violation-detection gap with 1.62 cm drone imagery?

Run AFTER preprocess_naip_sr.py.
Usage:
    python3 ~/hiz_pipeline/run_qwen25vl_naip_sr.py
"""

import csv
import gc
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import psutil
import torch
from PIL import Image
from tqdm import tqdm

PIPELINE_DIR   = Path.home() / "hiz_pipeline"
MODEL_ID       = "Qwen/Qwen2.5-VL-7B-Instruct"
MLX_MODEL_ID   = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
MODEL_PATH     = Path.home() / "hiz_data" / "models" / "qwen25vl-7b"
MLX_MODEL_PATH = Path.home() / "hiz_data" / "models" / "qwen25vl-7b-mlx"
TILES_DIR      = PIPELINE_DIR / "tiles_naip_sr"
MANIFEST_CSV   = TILES_DIR / "tile_manifest_naip_sr.csv"
RESULTS_DIR    = PIPELINE_DIR / "results" / "naip_sr_qwen25vl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))
from prompts import build_prompt, format_full_prompt_text, self_consistency_vote
from knowledge_graph.graph_rag_lookup import compute_risk_score

ZONE_PRIORITY   = {"Zone_0": 0, "Zone_1": 1, "Zone_2": 2}
IOU_THRESHOLD   = 0.5
MPS_FLUSH_EVERY = 5
MAX_NEW_TOKENS  = 512

# SR-NAIP specific preamble: tells model it's AI-upscaled public imagery
NAIP_SR_PREAMBLE = (
    "NOTE: This image is NAIP public aerial photography (originally 60 cm GSD) "
    "upscaled 4× to ~15 cm GSD using Real-ESRGAN AI super-resolution. "
    "High-frequency texture has been synthetically sharpened — fine details "
    "may appear clearer than in the original but are AI-reconstructed, not "
    "directly observed. Apply standard IBHS compliance detection. "
    "Note 'NAIP_SR' as imagery_source for all detections. "
    "Objects detectable at this scale: vehicles (>1 m), woodpiles, sheds, "
    "propane tanks (may be visible), vegetation patches.\n\n"
)


# ── Memory helpers ──────────────────────────────────────────────────────────

def sweep(label=""):
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if label:
        free = psutil.virtual_memory().available / 1e9
        print(f"  [MEM] {label}: {free:.1f} GB free")


# ── Model loading (same priority as run_qwen25vl.py) ───────────────────────

def ensure_mlx():
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        print("Installing mlx-vlm...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "mlx-vlm"],
            stdout=subprocess.DEVNULL,
        )


def _try_mlx():
    from mlx_vlm import load as mlx_load
    from mlx_vlm.utils import load_config
    src = str(MLX_MODEL_PATH) if (MLX_MODEL_PATH / "config.json").exists() \
          else MLX_MODEL_ID
    print(f"  MLX backend: loading from {src} ...")
    model, processor = mlx_load(src)
    processor._hiz_mlx_config = load_config(src)
    processor._hiz_mlx_src    = src
    print("  MLX model loaded (~4-bit, ~4 GB unified memory)")
    return model, processor, "mlx"


def _try_hf_mps():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    free = psutil.virtual_memory().available / 1e9
    if free < 8.0:
        raise RuntimeError(f"Only {free:.1f} GB free — need ≥8 GB for MPS")
    src = str(MODEL_PATH) if (MODEL_PATH / "config.json").exists() else MODEL_ID
    print(f"  HF/MPS backend: loading {src} ...")
    processor = AutoProcessor.from_pretrained(
        src, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).eval().to("mps")
    print("  HF/MPS float16 loaded.")
    return model, processor, "mps"


def _try_hf_cpu():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    src = str(MODEL_PATH) if (MODEL_PATH / "config.json").exists() else MODEL_ID
    print(f"  HF/CPU backend: loading {src} (slow) ...")
    processor = AutoProcessor.from_pretrained(src)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    print("  HF/CPU float16 loaded.")
    return model, processor, "cpu"


def load_model():
    sweep("before load")
    print("\nLoading Qwen2.5-VL-7B for SR-NAIP inference...")
    try:
        ensure_mlx()
        return _try_mlx()
    except Exception as e:
        print(f"  MLX failed ({type(e).__name__}: {e}). Trying HF/MPS...")
        sweep("after MLX fail")
    try:
        return _try_hf_mps()
    except Exception as e:
        print(f"  HF/MPS failed ({type(e).__name__}: {e}). Trying CPU...")
        sweep("after MPS fail")
    return _try_hf_cpu()


# ── Inference helpers ───────────────────────────────────────────────────────

def parse_json(raw: str):
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    s = raw.find("{")
    if s == -1:
        return None
    for e in range(len(raw), s, -1):
        try:
            return json.loads(raw[s:e])
        except Exception:
            continue
    return None


def iou(a, b):
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ub = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def deduplicate(dets):
    conf = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    kept = []
    for d in sorted(dets, key=lambda x: conf.get(x.get("confidence", "LOW"), 0),
                    reverse=True):
        bb = d.get("bounding_box_pixels", [])
        if not any(
            len(k.get("bounding_box_pixels", [])) == 4
            and iou(bb, k["bounding_box_pixels"]) > IOU_THRESHOLD
            for k in kept
        ):
            kept.append(d)
    return kept


def _infer_mlx(image, prompt_text, model, processor):
    import os, tempfile
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_chat_template
    from mlx_vlm.utils import load_config
    config = getattr(processor, "_hiz_mlx_config", None)
    if config is None:
        src = str(MLX_MODEL_PATH) if (MLX_MODEL_PATH / "config.json").exists() \
              else MLX_MODEL_ID
        config = load_config(src)
    formatted = mlx_chat_template(processor, config, prompt_text, num_images=1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = mlx_generate(
            model, processor, formatted, image=tmp_path,
            max_tokens=MAX_NEW_TOKENS, temperature=0.1, verbose=False,
        )
    finally:
        os.unlink(tmp_path)
    return result.text if hasattr(result, "text") else str(result)


def _infer_hf(image, prompt_text, model, processor, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt_text},
    ]}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_input], images=[image],
                       padding=True, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    del inputs
    raw = processor.batch_decode(
        gen_ids[:, prompt_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del gen_ids
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return raw


def infer_tile(tile_row, model, processor, device):
    parcel_id = tile_row["parcel_id"]
    meta = {
        "parcel_id": parcel_id,
        "site": tile_row.get("site", "unknown"),
        "gsd_cm": float(tile_row.get("gsd_cm", 15.0)),
        "chm_mean_in_tile": 0.0,
        "chm_max_in_tile":  0.0,
    }
    prompt_dict = build_prompt(
        tile_row["tile_path"], meta, tile_row["zone"], use_graph_rag=True
    )
    prompt_text = NAIP_SR_PREAMBLE + format_full_prompt_text(prompt_dict)
    image = Image.open(tile_row["tile_path"]).convert("RGB")

    raw = ""
    try:
        if device == "mlx":
            raw = _infer_mlx(image, prompt_text, model, processor)
        else:
            raw = _infer_hf(image, prompt_text, model, processor, device)
    except Exception as e:
        raw = f"[ERROR] {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        sweep()

    parsed = parse_json(raw)
    if parsed is None:
        return {"error": "JSON parse failed", "raw_output": raw,
                "detections": [], "overall_parcel_risk": "UNKNOWN",
                "aerial_limitations": ["SR_NAIP_PARSE_FAIL"], "parse_failed": True}
    parsed["raw_output"]   = raw
    parsed["parse_failed"] = False
    parsed.setdefault("detections", [])
    parsed.setdefault("overall_parcel_risk", "LOW")
    parsed.setdefault("aerial_limitations", [])
    # Tag source on every detection
    for d in parsed["detections"]:
        d["imagery_source"] = "NAIP_SR_4x"
    return parsed


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not MANIFEST_CSV.exists():
        sys.exit(
            f"SR manifest not found: {MANIFEST_CSV}\n"
            "Run: python3 ~/hiz_pipeline/preprocess_naip_sr.py first."
        )

    tiles = list(csv.DictReader(open(MANIFEST_CSV)))
    tiles.sort(key=lambda r: (ZONE_PRIORITY.get(r["zone"], 3), r["parcel_id"]))
    print(f"Loaded {len(tiles)} SR-NAIP tiles (~15 cm GSD, 2048×2048 px)")

    done = {f.stem for f in RESULTS_DIR.glob("*_r*_c*.json")}
    pending = [r for r in tiles
               if f"{Path(r['tile_path']).stem}" not in
               {f.stem for f in RESULTS_DIR.glob("*.json")}]
    # Use tile key matching
    done_keys = {f.stem for f in RESULTS_DIR.glob("*_r*_c*.json")}
    pending = [
        r for r in tiles
        if f"{r['parcel_id']}_r{r['row']}_c{r['col']}" not in done_keys
    ]
    if done_keys:
        print(f"Resuming: {len(done_keys)} done, {len(pending)} remaining.")
    if not pending:
        print("All SR-NAIP tiles processed.")
        return

    model, processor, device = load_model()
    print(f"  Device: {device}")
    sweep("after model load")

    # Benchmark
    print("\nBenchmarking on 3 tiles...")
    t0 = time.time()
    for tr in tiles[:3]:
        infer_tile(tr, model, processor, device)
    avg = (time.time() - t0) / 3
    est_h = avg * len(pending) / 3600
    print(f"  Avg/tile: {avg:.1f}s | ~{est_h:.1f}h for {len(pending)} tiles")
    if est_h > 1:
        print(f"  Estimated run time: {est_h:.1f}h — proceeding.")

    parcel_dets = defaultdict(list)
    parse_errs  = 0

    for i, tr in enumerate(tqdm(pending, desc="Qwen2.5-VL SR-NAIP", unit="tile")):
        pid  = tr["parcel_id"]
        ri, ci, zone = tr["row"], tr["col"], tr["zone"]
        out_path = RESULTS_DIR / f"{pid}_r{ri}_c{ci}.json"

        res = infer_tile(tr, model, processor, device)
        if res.get("parse_failed"):
            res = infer_tile(tr, model, processor, device)

        json.dump(
            {"tile_path": tr["tile_path"], "parcel_id": pid, "zone": zone,
             "gsd_cm": tr["gsd_cm"], "row": ri, "col": ci,
             "model": "Qwen2.5-VL-7B-Instruct",
             "imagery": "NAIP_SR_4x_RealESRGAN",
             "result": res},
            open(out_path, "w"), indent=2,
        )

        if res.get("parse_failed"):
            parse_errs += 1
        else:
            parcel_dets[pid].extend(res.get("detections", []))

        if (i + 1) % MPS_FLUSH_EVERY == 0:
            sweep(f"tile {i + 1}/{len(pending)}")

    # Parcel summaries
    for pid, dets in parcel_dets.items():
        deduped = deduplicate(dets)
        risk    = compute_risk_score(deduped)
        json.dump(
            {"parcel_id": pid, "model": "Qwen2.5-VL-7B-Instruct",
             "imagery": "NAIP_SR_4x_RealESRGAN",
             "total_detections": len(deduped), "detections": deduped, **risk},
            open(RESULTS_DIR / f"{pid}_summary.json", "w"), indent=2,
        )

    del model, processor
    sweep("after SR-NAIP inference complete")

    all_dets = [d for dl in parcel_dets.values() for d in dl]
    print(
        f"\n{'='*60}\n"
        f"  SR-NAIP INFERENCE DONE\n"
        f"  tiles={len(tiles)}  parse_errs={parse_errs}"
        f"  detections={len(all_dets)}\n"
        f"{'='*60}"
    )
    print(f"Results → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
