"""
run_geochat.py  (MPS-patched rewrite)
HIZ-VLM Pipeline — Step 6: GeoChat-7B Inference on Apple M4 MPS
PRIVACY: LOCAL INFERENCE ONLY — consented parcel orthomosaic data.
"""

import os, sys, gc, json, csv, time, re, traceback
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PIPELINE_DIR  = Path.home() / "hiz_pipeline"
GEOCHAT_REPO  = Path.home() / "hiz_data" / "GeoChat"
MODEL_PATH    = Path.home() / "hiz_data" / "models" / "geochat-7b"
TILES_DIR     = PIPELINE_DIR / "tiles"
RESULTS_DIR   = PIPELINE_DIR / "results" / "geochat"
MANIFEST_CSV  = TILES_DIR / "tile_manifest.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(GEOCHAT_REPO))

from prompts import build_prompt, format_full_prompt_text, self_consistency_vote
from knowledge_graph.graph_rag_lookup import compute_risk_score

ZONE_PRIORITY = {"Zone_0": 0, "Zone_1": 1, "Zone_2": 2}
IOU_THRESHOLD = 0.5
DEVICE = "mps"


def sweep():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def iou(a, b):
    if len(a) < 4 or len(b) < 4: return 0.0
    ix0,iy0 = max(a[0],b[0]), max(a[1],b[1])
    ix1,iy1 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix1-ix0)*max(0,iy1-iy0)
    ua = max(0,a[2]-a[0])*max(0,a[3]-a[1])
    ub = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(ua+ub-inter) if (ua+ub-inter)>0 else 0.0


def deduplicate(dets):
    conf = {"HIGH":2,"MEDIUM":1,"LOW":0}
    kept = []
    for d in sorted(dets, key=lambda x: conf.get(x.get("confidence","LOW"),0), reverse=True):
        bb = d.get("bounding_box_pixels",[])
        if not any(len(k.get("bounding_box_pixels",[]))==4 and
                   iou(bb, k["bounding_box_pixels"])>IOU_THRESHOLD for k in kept):
            kept.append(d)
    return kept


def parse_json(raw):
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m: raw = m.group(1).strip()
    s = raw.find("{")
    if s == -1: return None
    for e in range(len(raw), s, -1):
        try: return json.loads(raw[s:e])
        except: continue
    return None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_geochat():
    print(f"\n{'='*60}")
    print("  Loading GeoChat-7B on Apple M4 MPS")
    print(f"{'='*60}")
    from geochat.model.builder import load_pretrained_model
    from geochat.mm_utils import get_model_name_from_path
    model_name = get_model_name_from_path(str(MODEL_PATH))
    t0 = time.time()
    tok, model, img_proc, ctx_len = load_pretrained_model(
        str(MODEL_PATH), None, model_name, device="mps"
    )
    print(f"  Loaded in {time.time()-t0:.0f}s | ctx_len={ctx_len}")
    print(f"  Device: {next(model.parameters()).device}")
    return tok, model, img_proc, ctx_len


def unload_geochat(model, tok, img_proc):
    print("\nUnloading GeoChat from MPS...")
    del model, tok, img_proc
    sweep()
    print("  Done. MPS cache cleared.")


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_tile(tile_row, tok, model, img_proc):
    from geochat.mm_utils import (tokenizer_image_token, process_images,
                                   KeywordsStoppingCriteria)
    from geochat.constants  import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from geochat.conversation import conv_templates, SeparatorStyle

    parcel_id = tile_row["parcel_id"]
    zone      = tile_row["zone"]
    meta_path = TILES_DIR / f"{parcel_id}_meta.json"
    meta = json.load(open(meta_path))
    meta["chm_mean_in_tile"] = float(tile_row.get("chm_mean_in_tile", 0))
    meta["chm_max_in_tile"]  = float(tile_row.get("chm_max_in_tile",  0))

    prompt_dict = build_prompt(tile_row["tile_path"], meta, zone, use_graph_rag=True)
    prompt_text = format_full_prompt_text(prompt_dict)

    image = Image.open(tile_row["tile_path"]).convert("RGB")

    try:
        conv = conv_templates["llava_v1"].copy()
        inp  = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        input_ids = tokenizer_image_token(
            conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(DEVICE)

        img_tensor = process_images([image], img_proc, model.config)
        if isinstance(img_tensor, list):
            img_tensor = [t.to(DEVICE, dtype=torch.float16) for t in img_tensor]
        else:
            img_tensor = img_tensor.to(DEVICE, dtype=torch.float16)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        sc = KeywordsStoppingCriteria([stop_str], tok, input_ids)

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                images=img_tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[sc],
            )

        raw = tok.decode(out_ids[0, input_ids.shape[1]:],
                         skip_special_tokens=True).strip()
        del input_ids, img_tensor, out_ids
        sweep()

    except Exception as e:
        return {"error": str(e), "raw_output": "", "detections": [],
                "overall_parcel_risk": "UNKNOWN", "aerial_limitations": [],
                "parse_failed": True}

    parsed = parse_json(raw)
    if parsed is None:
        return {"error": "JSON parse failed", "raw_output": raw,
                "detections": [], "overall_parcel_risk": "UNKNOWN",
                "aerial_limitations": [], "parse_failed": True}
    parsed["raw_output"]   = raw
    parsed["parse_failed"] = False
    parsed.setdefault("detections", [])
    parsed.setdefault("overall_parcel_risk", "LOW")
    parsed.setdefault("aerial_limitations", [])
    return parsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not MANIFEST_CSV.exists():
        sys.exit(f"Manifest not found: {MANIFEST_CSV}. Run preprocess.py first.")

    tiles = list(csv.DictReader(open(MANIFEST_CSV)))
    tiles.sort(key=lambda r: (ZONE_PRIORITY.get(r["zone"],3), r["parcel_id"]))
    print(f"Loaded {len(tiles)} tiles | Zone_0:{sum(1 for t in tiles if t['zone']=='Zone_0')} "
          f"Zone_1:{sum(1 for t in tiles if t['zone']=='Zone_1')} "
          f"Zone_2:{sum(1 for t in tiles if t['zone']=='Zone_2')}")

    tok, model, img_proc, ctx_len = load_geochat()

    # Benchmark
    print("\nBenchmarking on 5 tiles...")
    t0 = time.time()
    for tr in tiles[:5]:
        infer_tile(tr, tok, model, img_proc)
    avg = (time.time()-t0)/5
    est_h = avg*len(tiles)/3600
    print(f"  Avg/tile: {avg:.1f}s | Estimated total: {est_h:.1f}h")
    if est_h > 3:
        print(f"  Estimated run time: {est_h:.1f}h — proceeding.")

    parcel_dets = defaultdict(list)
    parse_errs  = 0

    for i, tr in enumerate(tqdm(tiles, desc="GeoChat", unit="tile")):
        pid, ri, ci, zone = tr["parcel_id"], tr["row"], tr["col"], tr["zone"]

        res = infer_tile(tr, tok, model, img_proc)
        if res.get("parse_failed"):
            res = infer_tile(tr, tok, model, img_proc)   # one retry

        low_conf = [d for d in res.get("detections",[]) if d.get("confidence")=="LOW"]
        if low_conf:
            extras = [res] + [infer_tile(tr, tok, model, img_proc) for _ in range(2)]
            vote = self_consistency_vote(extras)
            res["detections"] = vote["voted_detections"]

        out = RESULTS_DIR / f"{pid}_r{ri}_c{ci}.json"
        json.dump({"tile_path":tr["tile_path"],"parcel_id":pid,"zone":zone,
                   "gsd_cm":tr["gsd_cm"],"row":ri,"col":ci,"result":res},
                  open(out,"w"), indent=2)

        if res.get("parse_failed"): parse_errs += 1
        else: parcel_dets[pid].extend(res.get("detections",[]))

        # Flush MPS every 50 tiles
        if (i+1) % 50 == 0:
            sweep()

    # Per-parcel summaries
    for pid, dets in parcel_dets.items():
        deduped = deduplicate(dets)
        risk    = compute_risk_score(deduped)
        json.dump({"parcel_id":pid,"model":"GeoChat-7B",
                   "total_detections":len(deduped),"detections":deduped,**risk},
                  open(RESULTS_DIR/f"{pid}_summary.json","w"), indent=2)

    unload_geochat(model, tok, img_proc)

    all_dets = [d for dl in parcel_dets.values() for d in dl]
    print(f"\n{'='*55}\n  GEOCHAT DONE | tiles={len(tiles)} errs={parse_errs} "
          f"dets={len(all_dets)}\n{'='*55}")
    for cls in ("vehicle","trash_can","propane_tank"):
        print(f"  {cls}: {sum(1 for d in all_dets if d.get('object_class')==cls)}")
    print(f"  Results → {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
