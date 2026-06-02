"""
run_ablation.py
HIZ-VLM Pipeline — Step 8: Ablation Study (No Graph-RAG Context)
Re-runs models on a 30% random sample of tiles using the plain
prompt (no regulatory context injection), to measure the impact of
Graph-RAG regulatory context on detection accuracy.

Models:
    qwen25vl  — Qwen2.5-VL-7B-Instruct (primary open-source model)
    internvl  — InternVL2-8B (secondary comparison model)
    geochat   — GeoChat-7B (legacy; skipped if transformers 5.x incompatible)

Usage:
    python3 ~/hiz_pipeline/run_ablation.py --model qwen25vl
    python3 ~/hiz_pipeline/run_ablation.py --model internvl
    python3 ~/hiz_pipeline/run_ablation.py           # all models sequentially

PRIVACY: LOCAL INFERENCE ONLY.
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch
from PIL import Image
from tqdm import tqdm

PIPELINE_DIR  = Path.home() / "hiz_pipeline"
TILES_DIR     = PIPELINE_DIR / "tiles"
MANIFEST_CSV  = TILES_DIR / "tile_manifest.csv"
ABL_QWEN      = PIPELINE_DIR / "results" / "qwen25vl_ablation"
ABL_INTERNVL  = PIPELINE_DIR / "results" / "internvl_ablation"
ABL_GEOCHAT   = PIPELINE_DIR / "results" / "geochat_ablation"
GEOCHAT_REPO  = Path.home() / "hiz_data" / "GeoChat"
GC_MODEL      = Path.home() / "hiz_data" / "models" / "geochat-7b"
IV_MODEL      = Path.home() / "hiz_data" / "models" / "internvl2-8b"
QW_MODEL      = Path.home() / "hiz_data" / "models" / "qwen25vl-7b"
QW_MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(GEOCHAT_REPO))

from prompts import build_prompt, format_full_prompt_text
ABL_QWEN.mkdir(parents=True, exist_ok=True)
ABL_INTERNVL.mkdir(parents=True, exist_ok=True)
ABL_GEOCHAT.mkdir(parents=True, exist_ok=True)

SAMPLE_FRACTION = 0.30
RANDOM_SEED     = 42


def load_sample() -> list[dict]:
    if not MANIFEST_CSV.exists():
        sys.exit(f"Manifest not found: {MANIFEST_CSV}")
    all_tiles = []
    with open(MANIFEST_CSV) as f:
        for row in csv.DictReader(f):
            all_tiles.append(row)
    random.seed(RANDOM_SEED)
    n = max(1, int(len(all_tiles) * SAMPLE_FRACTION))
    sample = random.sample(all_tiles, n)
    print(f"Ablation sample: {n} tiles ({SAMPLE_FRACTION*100:.0f}% of {len(all_tiles)})")
    return sample


def parse_json_response(raw: str) -> dict | None:
    import re
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m: raw = m.group(1).strip()
    start = raw.find("{")
    if start == -1: return None
    for end in range(len(raw), start, -1):
        try: return json.loads(raw[start:end])
        except json.JSONDecodeError: continue
    return None


def sweep_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─── Qwen2.5-VL ablation (no Graph-RAG) ──────────────────────────────────────

def run_qwen25vl_ablation(tiles: list[dict]):
    print("\n[ABLATION] Loading Qwen2.5-VL-7B for ablation (no Graph-RAG)...")
    sweep_memory()

    from transformers import (
        Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    )
    src = str(QW_MODEL) if (QW_MODEL / "config.json").exists() else QW_MODEL_ID
    processor = AutoProcessor.from_pretrained(
        src, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
    )
    model = None
    for quant, kwargs in [
        ("4-bit", dict(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"),
            device_map="auto", low_cpu_mem_usage=True)),
        ("8-bit", dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                       device_map="auto", low_cpu_mem_usage=True)),
        ("fp16-cpu", dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)),
    ]:
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                src, **kwargs).eval()
            device = next(model.parameters()).device.type
            print(f"  Loaded {quant} on {device}")
            break
        except Exception as e:
            print(f"  {quant} failed ({e}). Trying next...")
            sweep_memory()

    if model is None:
        print("[ABLATION] Could not load Qwen2.5-VL. Skipping.")
        return

    for tile_row in tqdm(tiles, desc="Qwen2.5-VL ablation", unit="tile"):
        parcel_id = tile_row["parcel_id"]
        meta_path = TILES_DIR / f"{parcel_id}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["chm_mean_in_tile"] = float(tile_row.get("chm_mean_in_tile", 0))
        meta["chm_max_in_tile"]  = float(tile_row.get("chm_max_in_tile",  0))

        # use_graph_rag=False is the ablation condition
        prompt_dict = build_prompt(
            tile_row["tile_path"], meta, tile_row["zone"], use_graph_rag=False
        )
        prompt_text = format_full_prompt_text(prompt_dict)
        image = Image.open(tile_row["tile_path"]).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_input], images=[image],
                padding=True, return_tensors="pt"
            )
            if device != "cpu":
                inputs = {k: v.to(device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            with torch.inference_mode():
                gen_ids = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            prompt_len = inputs["input_ids"].shape[1]
            raw = processor.batch_decode(
                gen_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            del inputs
        except Exception as e:
            raw = f"[ERROR] {e}"

        parsed = parse_json_response(raw) or {"detections": [], "raw": raw}
        out_path = ABL_QWEN / f"{parcel_id}_r{tile_row['row']}_c{tile_row['col']}.json"
        json.dump({
            "tile_path": tile_row["tile_path"],
            "parcel_id": parcel_id,
            "zone":      tile_row["zone"],
            "ablation":  True,
            "use_graph_rag": False,
            "model":     "Qwen2.5-VL-7B-Instruct",
            "result":    parsed,
            "raw":       raw,
        }, open(out_path, "w"), indent=2)

    del model, processor
    sweep_memory()
    print("[ABLATION] Qwen2.5-VL ablation complete.")


# ─── GeoChat ablation ─────────────────────────────────────────────────────────

def run_geochat_ablation(tiles: list[dict]):
    print("\n[ABLATION] Loading GeoChat-7B...")
    sys.path.insert(0, str(GEOCHAT_REPO))
    try:
        from geochat.model.builder import load_pretrained_model
        from geochat.mm_utils import (get_model_name_from_path,
                                       tokenizer_image_token,
                                       KeywordsStoppingCriteria)
        from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from geochat.conversation import conv_templates, SeparatorStyle
    except ImportError as e:
        print(f"GeoChat import failed: {e}. Skipping GeoChat ablation.")
        return

    model_name = get_model_name_from_path(str(GC_MODEL))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        str(GC_MODEL), None, model_name
    )
    model = model.to("mps").eval()

    for tile_row in tqdm(tiles, desc="GeoChat ablation", unit="tile"):
        parcel_id = tile_row["parcel_id"]
        meta_path = TILES_DIR / f"{parcel_id}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["chm_mean_in_tile"] = float(tile_row.get("chm_mean_in_tile", 0))
        meta["chm_max_in_tile"]  = float(tile_row.get("chm_max_in_tile",  0))

        prompt_dict = build_prompt(
            tile_row["tile_path"], meta, tile_row["zone"], use_graph_rag=False
        )
        prompt_text = format_full_prompt_text(prompt_dict)
        image = Image.open(tile_row["tile_path"]).convert("RGB")

        try:
            conv = conv_templates["geochat"].copy()
            inp  = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)

            input_ids = tokenizer_image_token(
                conv.get_prompt(), tokenizer,
                IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to("mps")
            img_tensor = image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"].half().to("mps")
            stop_str = (conv.sep if conv.sep_style != SeparatorStyle.TWO
                        else conv.sep2)
            sc = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                out_ids = model.generate(
                    input_ids, images=img_tensor,
                    do_sample=True, temperature=0.1,
                    max_new_tokens=512, use_cache=True,
                    stopping_criteria=[sc],
                )
            raw = tokenizer.decode(
                out_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            del input_ids, img_tensor
        except Exception as e:
            raw = f"[ERROR] {e}"

        parsed = parse_json_response(raw) or {"detections": [], "raw": raw}
        out_path = ABL_GEOCHAT / f"{parcel_id}_r{tile_row['row']}_c{tile_row['col']}.json"
        with open(out_path, "w") as f:
            json.dump({
                "tile_path": tile_row["tile_path"],
                "parcel_id": parcel_id,
                "zone":      tile_row["zone"],
                "ablation":  True,
                "result":    parsed,
                "raw":       raw,
            }, f, indent=2)

    del model, tokenizer, image_processor
    sweep_memory()
    print("[ABLATION] GeoChat ablation complete.")


# ─── InternVL2 ablation ───────────────────────────────────────────────────────

def run_internvl_ablation(tiles: list[dict]):
    print("\n[ABLATION] Loading InternVL2-8B (16GB mode)...")
    sweep_memory()

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        str(IV_MODEL), trust_remote_code=True, use_fast=False
    )

    ram_gb = psutil.virtual_memory().total / 1e9
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            str(IV_MODEL),
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        print("  Loaded with 4-bit quantization.")
    except Exception:
        model = AutoModel.from_pretrained(
            str(IV_MODEL),
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval().to("mps")
        print("  Loaded with float16 on MPS.")

    from torchvision import transforms
    IMG_SIZE  = 448
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    for tile_row in tqdm(tiles, desc="InternVL2 ablation", unit="tile"):
        parcel_id = tile_row["parcel_id"]
        meta_path = TILES_DIR / f"{parcel_id}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["chm_mean_in_tile"] = float(tile_row.get("chm_mean_in_tile", 0))
        meta["chm_max_in_tile"]  = float(tile_row.get("chm_max_in_tile",  0))

        prompt_dict = build_prompt(
            tile_row["tile_path"], meta, tile_row["zone"], use_graph_rag=False
        )
        prompt_text = format_full_prompt_text(prompt_dict)
        image = Image.open(tile_row["tile_path"]).convert("RGB")
        pv = transform(image).unsqueeze(0).to(torch.float16).to("mps")

        try:
            with torch.inference_mode():
                raw = model.chat(
                    tokenizer, pv, prompt_text,
                    generation_config={
                        "max_new_tokens": 512, "do_sample": True,
                        "temperature": 0.1,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                )
        except Exception as e:
            raw = f"[ERROR] {e}"
        finally:
            del pv
            gc.collect()

        parsed = parse_json_response(raw) or {"detections": [], "raw": raw}
        out_path = ABL_INTERNVL / f"{parcel_id}_r{tile_row['row']}_c{tile_row['col']}.json"
        with open(out_path, "w") as f:
            json.dump({
                "tile_path": tile_row["tile_path"],
                "parcel_id": parcel_id,
                "zone":      tile_row["zone"],
                "ablation":  True,
                "result":    parsed,
                "raw":       raw,
            }, f, indent=2)

    del model, tokenizer
    sweep_memory()
    print("[ABLATION] InternVL2 ablation complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen25vl", "internvl", "geochat", "all"],
        default="all",
        help="Which model to run ablation for (default: all sequentially)"
    )
    args = parser.parse_args()

    tiles = load_sample()

    if args.model in ("qwen25vl", "all"):
        run_qwen25vl_ablation(tiles)
        sweep_memory()

    if args.model in ("internvl", "all"):
        run_internvl_ablation(tiles)
        sweep_memory()

    if args.model in ("geochat", "all"):
        run_geochat_ablation(tiles)
        sweep_memory()

    print("\n[ABLATION] All ablation runs complete.")


if __name__ == "__main__":
    main()
