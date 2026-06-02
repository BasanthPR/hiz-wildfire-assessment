"""
run_qwen25vl.py  (MPS-optimized, 16 GB RAM)
HIZ-VLM Pipeline — Step 7b: Qwen2.5-VL-7B-Instruct Inference

Strongest open-source VLM (2025): outperforms GeoChat-7B and InternVL2-8B
on spatial reasoning, fine-grained detection, and document compliance.
Replaces GeoChat-7B as primary open-source model.

Loading priority:
  1. BitsAndBytes 4-bit quantized (~4.5 GB) on MPS
  2. BitsAndBytes 8-bit quantized (~8 GB) on MPS
  3. float16 on CPU (slow fallback)

PRIVACY: LOCAL INFERENCE ONLY — Henri's consented parcel data.
"""

import csv
import gc
import json
import os
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
TILES_DIR      = PIPELINE_DIR / "tiles"
RESULTS_DIR    = PIPELINE_DIR / "results" / "qwen25vl"
# Use filtered manifest if it exists, otherwise fall back to full manifest
_FILTERED   = TILES_DIR / "tile_manifest_filtered.csv"
_FULL       = TILES_DIR / "tile_manifest.csv"
MANIFEST_CSV = _FILTERED if _FILTERED.exists() else _FULL
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))
from prompts import build_prompt, format_full_prompt_text, self_consistency_vote
from knowledge_graph.graph_rag_lookup import compute_risk_score

ZONE_PRIORITY   = {"Zone_0": 0, "Zone_1": 1, "Zone_2": 2}
IOU_THRESHOLD   = 0.5
MPS_FLUSH_EVERY = 5
MAX_NEW_TOKENS  = 512


# ── Memory helpers ─────────────────────────────────────────────────────────────

def sweep(label=""):
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if label:
        free = psutil.virtual_memory().available / 1e9
        print(f"  [MEM] {label}: {free:.1f} GB free")


# ── Download weights ───────────────────────────────────────────────────────────

def download_qwen25vl():
    """Download Qwen2.5-VL-7B-Instruct weights if not present."""
    safetensors = list(MODEL_PATH.glob("*.safetensors"))
    if MODEL_PATH.exists() and safetensors:
        print(f"Qwen2.5-VL-7B weights present ({len(safetensors)} shards).")
        return

    free_gb = psutil.disk_usage(str(Path.home())).free / 1e9
    print(f"\nFree disk: {free_gb:.1f} GB | Qwen2.5-VL-7B needs ~15 GB")
    if free_gb < 16:
        sys.exit(f"Not enough disk space ({free_gb:.1f} GB). Need 16 GB free.")

    print(f"Downloading Qwen2.5-VL-7B-Instruct to {MODEL_PATH} ...")
    from huggingface_hub import snapshot_download
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_PATH),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.pt", "*.bin"],   # prefer safetensors
    )
    print("Download complete.")


# ── Model loading ──────────────────────────────────────────────────────────────

def _try_mlx():
    """
    Try MLX backend (Apple Silicon native, ~4 GB at 4-bit).
    Returns (model, processor, 'mlx') or raises ImportError.
    Config is loaded once here and stored on the processor object to avoid
    per-tile load_config() calls.
    """
    from mlx_vlm import load as mlx_load
    from mlx_vlm.utils import load_config

    src = str(MLX_MODEL_PATH) if (MLX_MODEL_PATH / "config.json").exists() \
          else MLX_MODEL_ID
    print(f"  MLX backend: loading from {src} ...")
    model, processor = mlx_load(src)
    # Cache config on the processor so _infer_mlx doesn't reload it per tile
    processor._hiz_mlx_config = load_config(src)
    processor._hiz_mlx_src    = src
    print(f"  MLX model loaded (~4-bit, ~4 GB unified memory)")
    return model, processor, "mlx"


def _try_hf_mps():
    """
    HuggingFace transformers on MPS (float16, ~15 GB — only if ≥14 GB free).
    Returns (model, processor, 'mps') or raises RuntimeError.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    free = psutil.virtual_memory().available / 1e9
    if free < 8.0:
        raise RuntimeError(f"Only {free:.1f} GB RAM free — need ≥8 GB for MPS float16")

    src = str(MODEL_PATH) if (MODEL_PATH / "config.json").exists() else MODEL_ID
    print(f"  HF/MPS backend: loading {src} ...")
    processor = AutoProcessor.from_pretrained(
        src, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("mps")
    print("  HF/MPS float16 loaded.")
    return model, processor, "mps"


def _try_hf_cpu():
    """Last-resort: float16 on CPU (~15 GB RAM, slow ~60 s/tile)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    src = str(MODEL_PATH) if (MODEL_PATH / "config.json").exists() else MODEL_ID
    print(f"  HF/CPU backend: loading {src} (slow) ...")
    processor = AutoProcessor.from_pretrained(src)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    print("  HF/CPU float16 loaded.")
    return model, processor, "cpu"


def ensure_mlx():
    """Install mlx-vlm if not present."""
    try:
        import mlx_vlm
    except ImportError:
        print("Installing mlx-vlm (Apple Silicon native VLM framework)...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "mlx-vlm"],
            stdout=subprocess.DEVNULL,
        )
        import mlx_vlm  # noqa: F401


def load_qwen25vl():
    """
    Load Qwen2.5-VL-7B with the best backend available.
    Priority:
      1. MLX 4-bit (Apple Silicon native, ~4 GB, fastest on M-series)
      2. HF transformers float16 on MPS (~15 GB, needs free RAM)
      3. HF transformers float16 on CPU (slow fallback)
    Returns (model, processor, backend_str).
    """
    sweep("before load")
    print(f"\nLoading Qwen2.5-VL-7B ...")

    # Priority 1: MLX (best for M-series Mac)
    try:
        ensure_mlx()
        return _try_mlx()
    except Exception as e:
        print(f"  MLX failed ({type(e).__name__}: {e}). Trying HF/MPS...")
        sweep("after MLX fail")

    # Priority 2: HF on MPS
    try:
        return _try_hf_mps()
    except Exception as e:
        print(f"  HF/MPS failed ({type(e).__name__}: {e}). Trying CPU...")
        sweep("after MPS fail")

    # Priority 3: CPU
    return _try_hf_cpu()


def unload_qwen25vl(model, processor):
    print("\nUnloading Qwen2.5-VL from memory...")
    del model, processor
    sweep("after unload")


# ── Inference ──────────────────────────────────────────────────────────────────

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
    for d in sorted(dets,
                    key=lambda x: conf.get(x.get("confidence", "LOW"), 0),
                    reverse=True):
        bb = d.get("bounding_box_pixels", [])
        if not any(
            len(k.get("bounding_box_pixels", [])) == 4
            and iou(bb, k["bounding_box_pixels"]) > IOU_THRESHOLD
            for k in kept
        ):
            kept.append(d)
    return kept


def _infer_mlx(image: Image.Image, prompt_text: str, model, processor) -> str:
    """MLX backend inference.
    mlx_vlm 0.4.x signature: generate(model, processor, prompt, image=path, ...)
    """
    import os
    import tempfile
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_chat_template
    from mlx_vlm.utils import load_config

    # Use cached config (loaded once in _try_mlx, not per tile)
    config = getattr(processor, "_hiz_mlx_config", None)
    if config is None:
        src = str(MLX_MODEL_PATH) if (MLX_MODEL_PATH / "config.json").exists() \
              else MLX_MODEL_ID
        config = load_config(src)
    # Build the formatted chat prompt (includes system message + examples + user turn)
    formatted_prompt = mlx_chat_template(
        processor, config, prompt_text, num_images=1
    )

    # Save image to temp file (mlx_vlm.generate needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = mlx_generate(
            model, processor,
            formatted_prompt,          # 3rd positional: prompt
            image=tmp_path,            # keyword: image path
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            verbose=False,
        )
    finally:
        os.unlink(tmp_path)

    # mlx_vlm.generate returns a GenerationResult object; .text is the string
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _infer_hf(image: Image.Image, prompt_text: str, model, processor,
              device: str) -> str:
    """HuggingFace transformers (MPS or CPU) backend inference."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input], images=[image], padding=True, return_tensors="pt"
    )
    if device != "cpu":
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: lower peak memory, deterministic
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    del inputs  # free device tensors before decode
    raw = processor.batch_decode(
        gen_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del gen_ids  # critical: free MPS tensor or it accumulates across tiles
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return raw


def infer_tile(tile_row: dict, model, processor, device: str) -> dict:
    """Run Qwen2.5-VL inference on a single tile."""
    parcel_id = tile_row["parcel_id"]
    meta_path = TILES_DIR / f"{parcel_id}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["chm_mean_in_tile"] = float(tile_row.get("chm_mean_in_tile", 0))
    meta["chm_max_in_tile"]  = float(tile_row.get("chm_max_in_tile", 0))

    prompt_dict = build_prompt(
        tile_row["tile_path"], meta, tile_row["zone"], use_graph_rag=True
    )
    prompt_text = format_full_prompt_text(prompt_dict)
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
        return {
            "error": "JSON parse failed",
            "raw_output": raw,
            "detections": [],
            "overall_parcel_risk": "UNKNOWN",
            "aerial_limitations": [],
            "parse_failed": True,
        }
    parsed["raw_output"]   = raw
    parsed["parse_failed"] = False
    parsed.setdefault("detections", [])
    parsed.setdefault("overall_parcel_risk", "LOW")
    parsed.setdefault("aerial_limitations", [])
    return parsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    download_qwen25vl()

    if not MANIFEST_CSV.exists():
        sys.exit(
            f"Manifest not found: {MANIFEST_CSV}\n"
            "Run preprocess.py first, then optionally filter_tiles.py."
        )

    tiles = list(csv.DictReader(open(MANIFEST_CSV)))
    tiles.sort(key=lambda r: (ZONE_PRIORITY.get(r["zone"], 3), r["parcel_id"]))
    filtered = "(filtered)" if "filtered" in MANIFEST_CSV.name else "(full)"
    print(f"Loaded {len(tiles)} tiles {filtered} across "
          f"{len(set(r['parcel_id'] for r in tiles))} parcels.")

    # Resume: skip tiles with existing results
    done = {f.stem for f in RESULTS_DIR.glob("*_r*_c*.json")}
    pending = [
        r for r in tiles
        if f"{r['parcel_id']}_r{r['row']}_c{r['col']}" not in done
    ]
    if done:
        print(f"Resuming: {len(done)} tiles done, {len(pending)} remaining.")
    tiles = pending

    if not tiles:
        print("All tiles already processed. Run evaluate.py to generate the report.")
        return

    model, processor, device = load_qwen25vl()
    print(f"  Device: {device}")
    sweep("after model load")

    # Benchmark
    print("\nBenchmarking on 3 tiles...")
    t0 = time.time()
    for tr in tiles[:3]:
        infer_tile(tr, model, processor, device)
    avg = (time.time() - t0) / 3
    est_h = avg * len(tiles) / 3600
    print(f"  Avg/tile: {avg:.1f}s | ~{est_h:.1f}h for {len(tiles)} tiles")
    if est_h > 5:
        print(f"  Estimated run time: {est_h:.1f}h — proceeding.")

    parcel_dets = defaultdict(list)
    parse_errs  = 0

    for i, tr in enumerate(tqdm(tiles, desc="Qwen2.5-VL", unit="tile")):
        pid = tr["parcel_id"]
        ri, ci, zone = tr["row"], tr["col"], tr["zone"]
        out_path = RESULTS_DIR / f"{pid}_r{ri}_c{ci}.json"

        res = infer_tile(tr, model, processor, device)

        # Retry once on parse failure
        if res.get("parse_failed"):
            res = infer_tile(tr, model, processor, device)

        # Self-consistency on LOW-confidence detections
        low_conf = [d for d in res.get("detections", [])
                    if d.get("confidence") == "LOW"]
        if low_conf:
            extras = [res] + [
                infer_tile(tr, model, processor, device) for _ in range(2)
            ]
            vote = self_consistency_vote(extras)
            res["detections"] = vote["voted_detections"]

        json.dump(
            {
                "tile_path":  tr["tile_path"],
                "parcel_id":  pid,
                "zone":       zone,
                "gsd_cm":     tr["gsd_cm"],
                "row":        ri,
                "col":        ci,
                "model":      "Qwen2.5-VL-7B-Instruct",
                "result":     res,
            },
            open(out_path, "w"),
            indent=2,
        )

        if res.get("parse_failed"):
            parse_errs += 1
        else:
            parcel_dets[pid].extend(res.get("detections", []))

        if (i + 1) % MPS_FLUSH_EVERY == 0:
            sweep(f"tile {i + 1}/{len(tiles)}")

    # Parcel-level summaries
    for pid, dets in parcel_dets.items():
        deduped = deduplicate(dets)
        risk    = compute_risk_score(deduped)
        json.dump(
            {
                "parcel_id":        pid,
                "model":            "Qwen2.5-VL-7B-Instruct",
                "total_detections": len(deduped),
                "detections":       deduped,
                **risk,
            },
            open(RESULTS_DIR / f"{pid}_summary.json", "w"),
            indent=2,
        )

    unload_qwen25vl(model, processor)

    all_dets = [d for dl in parcel_dets.values() for d in dl]
    print(
        f"\n{'='*60}\n"
        f"  QWEN2.5-VL DONE\n"
        f"  tiles={len(tiles)+len(done)}  parse_errs={parse_errs}"
        f"  detections={len(all_dets)}\n"
        f"{'='*60}"
    )
    for cls in ("vehicle", "trash_can", "propane_tank"):
        n = sum(1 for d in all_dets if d.get("object_class") == cls)
        print(f"  {cls}: {n}")
    print(f"\n  Results → {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
