"""
launch_labelstudio.py
=====================
Starts Label Studio, creates the HIZ pre-annotation project, and imports
the merged COCO JSON produced by preannotate_groundtruth.py.

Run once:
    /opt/miniconda3/bin/python3 launch_labelstudio.py

Then open http://localhost:8080 in your browser.
Credentials: admin@hiz.local / hizadmin123
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "preannotations" / "images"
COCO_JSON  = BASE_DIR / "preannotations" / "ground_truth_coco.json"
LS_DATA    = BASE_DIR / "preannotations" / "labelstudio_data"

LS_PORT    = 8081
LS_EMAIL   = "admin@hiz.local"
LS_PASS    = "hizadmin123"

# 33 HIZ object classes
OBJECT_CLASSES = [
    "woodpile", "furniture", "car", "rv", "above_ground_pool_or_hot_tub",
    "play_set", "pergola_gazebo", "garbage_bin", "boat", "propane",
    "storage_shed", "clutter", "planters", "fuel_breaks", "irrigation",
    "driveway", "welcome_mat", "address_sign", "fuel_or_flame_wick", "hoses",
    "broom", "ladder", "portable_gas_pump", "curtains", "lights",
    "live_herb", "live_shrub", "live_tree", "dead_vegetation", "mulch",
    "deck_patio", "fence", "bbq_grill",
]

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" showInline="true">
""" + "\n".join(
    f'    <Label value="{cls}" background="#{abs(hash(cls)) % 0xFFFFFF:06X}"/>'
    for cls in OBJECT_CLASSES
) + """
  </RectangleLabels>
</View>"""


MIN_SCORE = 0.25   # filters ~37% low-confidence noise; preserves recall for rare aerial classes


def build_ls_tasks(coco_path: Path, images_dir: Path) -> list:
    """Convert merged COCO JSON → Label Studio task list with pre-annotations."""
    with open(coco_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image_id
    anns_by_img: dict = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    tasks = []
    for img in coco["images"]:
        img_path = Path(img["file_name"])
        # Label Studio needs a URI it can serve. We use /data/local-files/?d=...
        # with the absolute path — works when local file serving is enabled.
        rel = img_path.name  # just the filename; LS local storage uses the dir
        ls_uri = f"/data/local-files/?d={img_path}"

        W, H = img["width"], img["height"]

        predictions = []
        for ann in anns_by_img.get(img["id"], []):
            if ann.get("score", 1.0) < MIN_SCORE:
                continue
            x, y, w, h = ann["bbox"]   # COCO: x,y = top-left corner
            # Label Studio expects percentages relative to image size
            predictions.append({
                "from_name": "label",
                "to_name"  : "image",
                "type"     : "rectanglelabels",
                "value": {
                    "x"     : x / W * 100,
                    "y"     : y / H * 100,
                    "width" : w / W * 100,
                    "height": h / H * 100,
                    "rotation": 0,
                    "rectanglelabels": [cat_map[ann["category_id"]]],
                },
                "score": ann.get("score", 1.0),
            })

        tasks.append({
            "data": {
                "image"     : ls_uri,
                "parcel_id" : img.get("parcel_id", ""),
                "tile_row"  : img.get("tile_row", 0),
                "tile_col"  : img.get("tile_col", 0),
            },
            "predictions": [{"result": predictions, "score": 0.0, "model_version": "owlv2-base-patch16-ensemble"}] if predictions else [],
        })

    return tasks


def main():
    LS_DATA.mkdir(parents=True, exist_ok=True)

    # ── 1. Start Label Studio in the background ───────────────────────────────
    print("Starting Label Studio …")
    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"]   = str(IMAGES_DIR)
    env["LABEL_STUDIO_BASE_DATA_DIR"]               = str(LS_DATA)
    env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "true"

    ls_bin = Path(sys.executable).parent / "label-studio"
    ls_proc = subprocess.Popen(
        [str(ls_bin), "start",
         "--port", str(LS_PORT),
         "--username", LS_EMAIL,
         "--password", LS_PASS,
         "--no-browser",
         "--log-level", "WARNING",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the server to be ready
    import urllib.request
    url = f"http://localhost:{LS_PORT}/health"
    for i in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(url, timeout=3)
            break
        except Exception:
            print(f"  waiting … ({i*2}s)", end="\r")
    else:
        print("Label Studio did not start in 120s. Check logs.")
        ls_proc.terminate()
        sys.exit(1)

    print(f"\nLabel Studio is up at http://localhost:{LS_PORT}")

    # ── 2. Create project via SDK ─────────────────────────────────────────────
    try:
        from label_studio_sdk import Client
    except ImportError:
        print("label-studio-sdk not found — run: pip install label-studio-sdk")
        ls_proc.wait()
        return

    ls = Client(url=f"http://localhost:{LS_PORT}", api_key=None)
    ls.check_connection()

    # Authenticate and get API key
    import urllib.parse, urllib.request as ur
    login_data = urllib.parse.urlencode({"email": LS_EMAIL, "password": LS_PASS}).encode()
    req = ur.Request(f"http://localhost:{LS_PORT}/api/token/", data=login_data,
                     headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        resp = json.loads(ur.urlopen(req).read())
        token = resp.get("token") or resp.get("access")
    except Exception:
        # Try the DRF token endpoint
        import http.client, json as _json
        conn = http.client.HTTPConnection("localhost", LS_PORT)
        conn.request("POST", "/api/token/",
                     json.dumps({"username": LS_EMAIL, "password": LS_PASS}),
                     {"Content-Type": "application/json"})
        resp = _json.loads(conn.getresponse().read())
        token = resp.get("token") or resp.get("access")

    ls = Client(url=f"http://localhost:{LS_PORT}", api_key=token)

    project = ls.start_project(
        title="HIZ Pre-Annotations (OWLv2)",
        label_config=LABEL_CONFIG,
    )
    print(f"Created project #{project.id}: HIZ Pre-Annotations (OWLv2)")

    # ── 3. Convert COCO → LS tasks and import ────────────────────────────────
    print("Converting COCO JSON → Label Studio tasks …")
    tasks = build_ls_tasks(COCO_JSON, IMAGES_DIR)
    print(f"  {len(tasks)} tasks to import")

    BATCH = 500
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i:i + BATCH]
        project.import_tasks(batch)
        print(f"  imported {min(i + BATCH, len(tasks))}/{len(tasks)}", end="\r")

    print(f"\nDone. {len(tasks)} tiles loaded with OWLv2 pre-annotations.")
    print(f"\nOpen: http://localhost:{LS_PORT}/projects/{project.id}/")
    print(f"Login: {LS_EMAIL} / {LS_PASS}")
    print("\nPress Ctrl+C to stop Label Studio.")

    try:
        ls_proc.wait()
    except KeyboardInterrupt:
        ls_proc.terminate()
        print("\nLabel Studio stopped.")


if __name__ == "__main__":
    main()
