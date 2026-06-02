"""
run_qwen25vl_naip.py
HIZ-VLM Pipeline — Public Imagery Inference (NAIP)

Runs Qwen2.5-VL-7B on publicly-available NAIP tiles (60 cm GSD).
Uses same Graph-RAG + Geo-CoT prompt as run_qwen25vl.py but with
a NAIP-specific preamble noting the coarser resolution.

Results stored in ~/hiz_pipeline/results/naip_qwen25vl/

Usage:
    python3 ~/hiz_pipeline/run_qwen25vl_naip.py
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

PIPELINE_DIR = Path.home() / "hiz_pipeline"
MODEL_PATH   = Path.home() / "hiz_data" / "models" / "qwen25vl-7b"
MODEL_ID     = "Qwen/Qwen2.5-VL-7B-Instruct"
TILES_DIR    = PIPELINE_DIR / "tiles_naip"
MANIFEST_CSV = TILES_DIR / "tile_manifest_naip.csv"
RESULTS_DIR  = PIPELINE_DIR / "results" / "naip_qwen25vl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))
from prompts import build_prompt, format_full_prompt_text, self_consistency_vote
from knowledge_graph.graph_rag_lookup import compute_risk_score, get_regulatory_context

ZONE_PRIORITY   = {"Zone_0": 0, "Zone_1": 1, "Zone_2": 2}
IOU_THRESHOLD   = 0.5
MPS_FLUSH_EVERY = 15
MAX_NEW_TOKENS  = 512

# NAIP-specific prompt note (prepended to standard prompt)
NAIP_PREAMBLE = (
    "NOTE: This image is from NAIP (National Agriculture Imagery Program) "
    "public aerial photography at approximately 60 cm ground sampling distance "
    "(GSD). Fine structural details visible in drone imagery (e.g., individual "
    "propane tank valves, small woodpiles) may not be resolved. Focus detection "
    "on objects resolvable at this scale: vehicles (>2 m), large woodpiles "
    ">1 m), storage sheds (>2 m), and dense dry vegetation patches. "
    "Apply the same regulatory standards but note reduced confidence for "
    "objects smaller than 2-3 pixels.\n\n"
)


def sweep(label=""):
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if label:
        free = psutil.virtual_memory().available / 1e9
        print(f"  [MEM] {label}: {free:.1f} GB free")


def load_qwen25vl():
    from transformers import (
        Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    )
    src = str(MODEL_PATH) if (MODEL_PATH / "config.json").exists() else MODEL_ID
    sweep("before load")
    processor = AutoProcessor.from_pretrained(
        src, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
    )
    for quant, kwargs in [
        ("4-bit", dict(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
        ), device_map="auto", low_cpu_mem_usage=True)),
        ("8-bit", dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                       device_map="auto", low_cpu_mem_usage=True)),
        ("fp16-cpu", dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)),
    ]:
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                src, **kwargs
            ).eval()
            device = next(model.parameters()).device.type
            print(f"  Loaded {quant} on {device}")
            return model, processor, device
        except Exception as e:
            print(f"  {quant} failed ({e}). Trying next...")
            sweep(f"after {quant} fail")
    sys.exit("Could not load Qwen2.5-VL in any quantization mode.")


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


def infer_tile_naip(tile_row: dict, model, processor, device: str) -> dict:
    parcel_id = tile_row["parcel_id"]
    zone      = tile_row["zone"]

    # Build prompt with NAIP preamble
    meta = {
        "parcel_id": parcel_id,
        "site": tile_row.get("site", "unknown"),
        "gsd_cm": float(tile_row.get("gsd_cm", 60.0)),
        "chm_mean_in_tile": 0.0,
        "chm_max_in_tile": 0.0,
    }
    prompt_dict  = build_prompt(tile_row["tile_path"], meta, zone, use_graph_rag=True)
    base_prompt  = format_full_prompt_text(prompt_dict)
    prompt_text  = NAIP_PREAMBLE + base_prompt

    image = Image.open(tile_row["tile_path"]).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]

    try:
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        if device != "cpu":
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.1,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        raw = processor.batch_decode(
            gen_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    except Exception as e:
        raw = f"[ERROR] {type(e).__name__}: {e}"
    finally:
        del inputs
        sweep()

    parsed = parse_json(raw)
    if parsed is None:
        return {"error": "JSON parse failed", "raw_output": raw,
                "detections": [], "overall_parcel_risk": "UNKNOWN",
                "aerial_limitations": ["NAIP_LOW_RESOLUTION"],
                "parse_failed": True}
    parsed["raw_output"]   = raw
    parsed["parse_failed"] = False
    parsed.setdefault("detections", [])
    parsed.setdefault("overall_parcel_risk", "LOW")
    parsed.setdefault("aerial_limitations", ["NAIP_60CM_GSD"])
    # Tag each detection as coming from public imagery
    for d in parsed["detections"]:
        d["imagery_source"] = "NAIP_public"
    return parsed


def main():
    if not MANIFEST_CSV.exists():
        sys.exit(
            f"NAIP manifest not found: {MANIFEST_CSV}\n"
            "Run: python3 ~/hiz_pipeline/download_public_imagery.py\n"
            "     python3 ~/hiz_pipeline/preprocess_naip.py"
        )

    tiles = list(csv.DictReader(open(MANIFEST_CSV)))
    tiles.sort(key=lambda r: (ZONE_PRIORITY.get(r["zone"], 3), r["parcel_id"]))
    print(f"Loaded {len(tiles)} NAIP tiles (public imagery)")

    done = {f.stem for f in RESULTS_DIR.glob("*_r*_c*.json")}
    pending = [r for r in tiles
               if f"{r['parcel_id']}_r{r['row']}_c{r['col']}" not in done]
    if done:
        print(f"Resuming: {len(done)} done, {len(pending)} remaining.")
    tiles = pending

    if not tiles:
        print("All NAIP tiles processed.")
        return

    model, processor, device = load_qwen25vl()
    sweep("after NAIP model load")

    parcel_dets = defaultdict(list)
    parse_errs  = 0

    for i, tr in enumerate(tqdm(tiles, desc="Qwen2.5-VL NAIP", unit="tile")):
        pid = tr["parcel_id"]
        ri, ci, zone = tr["row"], tr["col"], tr["zone"]

        res = infer_tile_naip(tr, model, processor, device)
        if res.get("parse_failed"):
            res = infer_tile_naip(tr, model, processor, device)

        json.dump(
            {"tile_path": tr["tile_path"], "parcel_id": pid, "zone": zone,
             "gsd_cm": tr["gsd_cm"], "row": ri, "col": ci,
             "model": "Qwen2.5-VL-7B-Instruct", "imagery": "NAIP",
             "result": res},
            open(RESULTS_DIR / f"{pid}_r{ri}_c{ci}.json", "w"),
            indent=2,
        )

        if res.get("parse_failed"):
            parse_errs += 1
        else:
            parcel_dets[pid].extend(res.get("detections", []))

        if (i + 1) % MPS_FLUSH_EVERY == 0:
            sweep(f"tile {i + 1}/{len(tiles)}")

    for pid, dets in parcel_dets.items():
        deduped = deduplicate(dets)
        risk    = compute_risk_score(deduped)
        json.dump(
            {"parcel_id": pid, "model": "Qwen2.5-VL-7B-Instruct",
             "imagery": "NAIP_public", "total_detections": len(deduped),
             "detections": deduped, **risk},
            open(RESULTS_DIR / f"{pid}_summary.json", "w"),
            indent=2,
        )

    del model, processor
    sweep("after NAIP inference complete")

    all_dets = [d for dl in parcel_dets.values() for d in dl]
    print(f"\nNAIP Inference Complete | tiles={len(tiles)+len(done)}"
          f"  errs={parse_errs}  detections={len(all_dets)}")
    print(f"Results → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
