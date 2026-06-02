"""
run_internvl.py  (MPS-patched, 16 GB RAM, sequential)
HIZ-VLM Pipeline — Step 7: InternVL2-8B Inference
Downloads InternVL2-8B if not present, then runs inference.
Must be run AFTER run_geochat.py has fully exited.
PRIVACY: LOCAL INFERENCE ONLY.
"""

import os, sys, gc, json, csv, re, time, subprocess, traceback
from pathlib import Path
from collections import defaultdict

import psutil, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

PIPELINE_DIR = Path.home() / "hiz_pipeline"
MODEL_PATH   = Path.home() / "hiz_data" / "models" / "internvl2-8b"
TILES_DIR    = PIPELINE_DIR / "tiles"
RESULTS_DIR  = PIPELINE_DIR / "results" / "internvl"
MANIFEST_CSV = TILES_DIR / "tile_manifest.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))
from prompts import build_prompt, format_full_prompt_text, self_consistency_vote
from knowledge_graph.graph_rag_lookup import compute_risk_score

ZONE_PRIORITY   = {"Zone_0":0,"Zone_1":1,"Zone_2":2}
IOU_THRESHOLD   = 0.5
MPS_FLUSH_EVERY = 20
DEVICE          = "mps"

# ── Memory helpers ────────────────────────────────────────────────────────────

def sweep(label=""):
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if label:
        free = psutil.virtual_memory().available/1e9
        print(f"  [MEM] {label}: {free:.1f} GB free")


# ── Download InternVL2-8B (called only if weights missing) ───────────────────

def download_internvl():
    """
    Download InternVL2-8B from HuggingFace, freeing GeoChat weights first
    if disk space is tight.
    """
    if MODEL_PATH.exists() and any(MODEL_PATH.glob("*.safetensors")):
        print("InternVL2-8B weights already present.")
        return

    # Check disk space
    free_gb = psutil.disk_usage(str(Path.home())).free / 1e9
    print(f"\nFree disk: {free_gb:.1f} GB | InternVL2-8B needs ~17 GB")

    if free_gb < 17:
        # Offer to delete GeoChat weights to free space
        gc_dir = Path.home() / "hiz_data" / "models" / "geochat-7b"
        gc_bins = list(gc_dir.glob("*.bin"))
        gc_size = sum(f.stat().st_size for f in gc_bins) / 1e9
        print(f"\n  GeoChat weights ({gc_size:.1f} GB) can be deleted since")
        print("  GeoChat inference is complete and results are saved.")
        for f in gc_bins:
            f.unlink()
            print(f"  Deleted: {f.name}")
            sweep("after GeoChat cleanup")
            free_gb = psutil.disk_usage(str(Path.home())).free / 1e9
            print(f"  Free disk now: {free_gb:.1f} GB")
        else:
            print("  Proceeding with download anyway (may fail if disk fills up).")

    print(f"\nDownloading InternVL2-8B (~16 GB)...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="OpenGVLab/InternVL2-8B",
        local_dir=str(MODEL_PATH),
        local_dir_use_symlinks=False,
    )
    print("InternVL2-8B download complete.")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_internvl2():
    """
    16 GB RAM strategy:
      Priority 1 — 4-bit NF4 via bitsandbytes (~4 GB)
      Priority 2 — float16 on MPS (~16 GB, may OOM)
      Priority 3 — float16 on CPU (slow, last resort)
    """
    from transformers import AutoTokenizer, AutoModel

    sweep("before model load")
    print(f"\n{'='*60}")
    print("  Loading InternVL2-8B  |  16 GB RAM MODE")
    print(f"{'='*60}")
    print(f"  Free RAM: {psutil.virtual_memory().available/1e9:.1f} GB")

    # Block if GeoChat process is still running
    try:
        r = subprocess.run(["pgrep","-f","run_geochat"], capture_output=True, text=True)
        if r.stdout.strip():
            sys.exit("[ERROR] run_geochat.py is still running. Wait for it to finish first.")
    except FileNotFoundError:
        pass

    tok = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, use_fast=False
    )
    print("  Tokenizer: OK")

    # Priority 1: 4-bit NF4
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            str(MODEL_PATH), quantization_config=bnb,
            torch_dtype=torch.float16, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        print("  Loaded with 4-bit NF4 (~4 GB).")
        return model, tok, "bnb4bit"
    except Exception as e:
        print(f"  4-bit failed ({type(e).__name__}). Trying float16 MPS...")
        sweep("after failed 4-bit")

    # Priority 2: float16 on MPS
    try:
        model = AutoModel.from_pretrained(
            str(MODEL_PATH), torch_dtype=torch.float16,
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).eval().to("mps")
        print("  Loaded float16 on MPS.")
        return model, tok, "mps"
    except (RuntimeError, Exception) as e:
        if "out of memory" in str(e).lower() or "alloc" in str(e).lower():
            print(f"  MPS OOM. Falling back to CPU...")
            sweep("after MPS OOM")
        else:
            raise

    # Priority 3: CPU
    model = AutoModel.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval()
    print("  Loaded on CPU (slow).")
    return model, tok, "cpu"


def unload_internvl(model, tok):
    print("\nUnloading InternVL2 from memory...")
    del model, tok
    sweep("after unload")


# ── Image preprocessing for InternVL2 ────────────────────────────────────────

IMG_SIZE  = 448
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def make_pixel_values(image: Image.Image, device: str) -> torch.Tensor:
    pv = _transform(image).unsqueeze(0).to(torch.float16)
    if device == "mps":
        pv = pv.to("mps")
    return pv


# ── Inference ─────────────────────────────────────────────────────────────────

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


def iou(a,b):
    if len(a)<4 or len(b)<4: return 0.0
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1])
    ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)
    ua=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    ub=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(ua+ub-inter) if ua+ub-inter>0 else 0.0


def deduplicate(dets):
    conf={"HIGH":2,"MEDIUM":1,"LOW":0}
    kept=[]
    for d in sorted(dets,key=lambda x:conf.get(x.get("confidence","LOW"),0),reverse=True):
        bb=d.get("bounding_box_pixels",[])
        if not any(len(k.get("bounding_box_pixels",[]))==4 and
                   iou(bb,k["bounding_box_pixels"])>IOU_THRESHOLD for k in kept):
            kept.append(d)
    return kept


def infer_tile(tile_row, model, tok, device):
    parcel_id = tile_row["parcel_id"]
    meta = json.load(open(TILES_DIR / f"{parcel_id}_meta.json"))
    meta["chm_mean_in_tile"] = float(tile_row.get("chm_mean_in_tile", 0))
    meta["chm_max_in_tile"]  = float(tile_row.get("chm_max_in_tile",  0))

    prompt_dict = build_prompt(tile_row["tile_path"], meta,
                               tile_row["zone"], use_graph_rag=True)
    prompt_text = format_full_prompt_text(prompt_dict)
    image = Image.open(tile_row["tile_path"]).convert("RGB")
    pv    = make_pixel_values(image, device)

    gen_cfg = dict(max_new_tokens=512, do_sample=True, temperature=0.1,
                   pad_token_id=tok.eos_token_id)
    try:
        with torch.inference_mode():
            raw = model.chat(tok, pv, prompt_text, generation_config=gen_cfg)
        raw = raw if isinstance(raw, str) else str(raw)
    except Exception as e:
        raw = f"[ERROR] {e}"
    finally:
        del pv
        gc.collect()

    parsed = parse_json(raw)
    if parsed is None:
        return {"error":"JSON parse failed","raw_output":raw,"detections":[],
                "overall_parcel_risk":"UNKNOWN","aerial_limitations":[],
                "parse_failed":True}
    parsed["raw_output"]   = raw
    parsed["parse_failed"] = False
    parsed.setdefault("detections",[])
    parsed.setdefault("overall_parcel_risk","LOW")
    parsed.setdefault("aerial_limitations",[])
    return parsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Download weights if needed (also handles GeoChat cleanup prompt)
    download_internvl()

    if not MANIFEST_CSV.exists():
        sys.exit(f"Manifest not found: {MANIFEST_CSV}")

    tiles = list(csv.DictReader(open(MANIFEST_CSV)))
    tiles.sort(key=lambda r: (ZONE_PRIORITY.get(r["zone"],3), r["parcel_id"]))
    print(f"Loaded {len(tiles)} tiles.")

    model, tok, device = load_internvl2()
    print(f"  Device: {device}")

    # Benchmark
    print("\nBenchmarking 5 tiles...")
    t0 = time.time()
    for tr in tiles[:5]:
        infer_tile(tr, model, tok, device)
    avg = (time.time()-t0)/5
    est_h = avg*len(tiles)/3600
    print(f"  Avg/tile: {avg:.1f}s | Estimated: {est_h:.1f}h")
    if est_h > 3:
        print(f"  Estimated run time: {est_h:.1f}h — proceeding.")

    parcel_dets = defaultdict(list)
    parse_errs  = 0

    for i, tr in enumerate(tqdm(tiles, desc="InternVL2", unit="tile")):
        pid, ri, ci, zone = tr["parcel_id"], tr["row"], tr["col"], tr["zone"]

        res = infer_tile(tr, model, tok, device)
        if res.get("parse_failed"):
            res = infer_tile(tr, model, tok, device)

        low_conf = [d for d in res.get("detections",[]) if d.get("confidence")=="LOW"]
        if low_conf:
            extras = [res] + [infer_tile(tr, model, tok, device) for _ in range(2)]
            vote = self_consistency_vote(extras)
            res["detections"] = vote["voted_detections"]

        json.dump({"tile_path":tr["tile_path"],"parcel_id":pid,"zone":zone,
                   "gsd_cm":tr["gsd_cm"],"row":ri,"col":ci,"result":res},
                  open(RESULTS_DIR/f"{pid}_r{ri}_c{ci}.json","w"), indent=2)

        if res.get("parse_failed"): parse_errs += 1
        else: parcel_dets[pid].extend(res.get("detections",[]))

        if (i+1) % MPS_FLUSH_EVERY == 0:
            sweep(f"tile {i+1}")

    for pid, dets in parcel_dets.items():
        deduped = deduplicate(dets)
        risk    = compute_risk_score(deduped)
        json.dump({"parcel_id":pid,"model":"InternVL2-8B",
                   "total_detections":len(deduped),"detections":deduped,**risk},
                  open(RESULTS_DIR/f"{pid}_summary.json","w"), indent=2)

    unload_internvl(model, tok)

    all_dets = [d for dl in parcel_dets.values() for d in dl]
    print(f"\n{'='*55}\n  INTERNVL2 DONE | tiles={len(tiles)} errs={parse_errs} "
          f"dets={len(all_dets)}\n{'='*55}")
    for cls in ("vehicle","trash_can","propane_tank"):
        print(f"  {cls}: {sum(1 for d in all_dets if d.get('object_class')==cls)}")
    print(f"  Results → {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
