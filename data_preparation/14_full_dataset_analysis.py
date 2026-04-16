"""
14_full_dataset_analysis.py
===========================
Full pipeline on the complete YourGround dataset (3,078 reports, 3,008 unique locations):

  1. Download front+back Street View images for every location (skip cached)
  2. Run Gemini Vision on each location (skip cached)
  3. Merge 29 visual features into the full dataset
  4. Output ML-ready CSV with all original columns + visual features + engineered targets
  5. Print correlation analysis

Outputs
-------
  report_images/              – Street View images (N + S per location)
  cache_llm_report/           – Gemini response cache (JSON per location)
  yourground_ml_features.csv  – Full ML-ready dataset
  full_llm_report.txt         – Correlation report

Usage
-----
  python 14_full_dataset_analysis.py                # full run
  python 14_full_dataset_analysis.py --skip-sv      # skip image download
  python 14_full_dataset_analysis.py --skip-llm     # skip Gemini, rebuild CSV only
  python 14_full_dataset_analysis.py --sample 50    # test on 50 locations
"""

import argparse
import base64
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YOURGROUND_CSV = Path(r"C:/Users/papad/Desktop/Iaacthesis/yourground_data2.csv")
IMAGE_DIR      = Path("report_images")
CACHE_DIR      = Path("cache_llm_report")
OUTPUT_CSV     = Path("yourground_ml_features.csv")
OUTPUT_REPORT  = Path("full_llm_report.txt")
API_TXT        = Path("api.txt")

SV_SIZE        = "640x400"
SV_FOV         = 90
SV_PITCH       = 0
SV_DELAY       = 0.25     # seconds between Street View calls
GEMINI_DELAY   = 8        # seconds between Gemini calls
GEMINI_MODELS  = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

SCORE_KEYS = [
    "vividness","brightness","colour_warmth","visual_complexity",
    "people_present","activity_level","greenery","tree_canopy",
    "lighting_infrastructure","lighting_quality","cctv_visible",
    "benches_seating","fencing_barriers","maintenance_level",
    "graffiti","garbage_litter","unkempt_vegetation",
    "concrete_dominance","brick_surfaces","glass_facades",
    "asphalt_path","natural_surfaces","enclosure","visibility",
    "active_frontage","blank_walls","road_traffic",
    "perceived_safety","perceived_upkeep",
]

ANALYSIS_PROMPT = """\
You are an urban environment analyst. You will receive one or two street-level \
photographs of the same location (front-facing and back-facing views).

Analyse the combined scene and score each feature from 0.0 to 1.0.
Where two images are provided, average your impression across both.

FEATURES TO SCORE (all 0.0-1.0):

Visual atmosphere
  vividness          - colour saturation and visual richness (0=dull/grey, 1=vibrant)
  brightness         - overall image brightness/exposure (0=dark, 1=bright)
  colour_warmth      - dominant colour warmth (0=cold blues/greys, 1=warm yellows/reds)
  visual_complexity  - density of visual elements (0=bare/empty, 1=highly complex)

People & activity
  people_present     - any pedestrians visible (0=none, 1=many)
  activity_level     - sense of life/movement (0=dead, 1=busy)

Vegetation & nature
  greenery           - trees, shrubs, maintained vegetation (0=none, 1=abundant)
  tree_canopy        - overhead canopy cover (0=none, 1=full)

Infrastructure & lighting
  lighting_infrastructure - visible lamp posts / street lights (0=none, 1=many)
  lighting_quality   - how well-lit the space feels (0=dark, 1=bright)
  cctv_visible       - surveillance cameras visible (0=none, 1=yes)
  benches_seating    - public seating visible (0=none, 1=yes)
  fencing_barriers   - fences, walls, barriers (0=none, 1=dominant)

Maintenance & disorder
  maintenance_level  - upkeep of surfaces and buildings (0=very neglected, 1=pristine)
  graffiti           - tags, vandalism, unauthorized markings (0=none, 1=heavy)
  garbage_litter     - visible rubbish, overflowing bins, debris (0=none, 1=heavy)
  unkempt_vegetation - overgrown, dying, or messy plants (0=none, 1=dominant)

Materials & surfaces
  concrete_dominance - concrete/brutalist surfaces dominate (0=none, 1=dominant)
  brick_surfaces     - brick walls or paths (0=none, 1=dominant)
  glass_facades      - glass shopfronts or buildings (0=none, 1=dominant)
  asphalt_path       - asphalt road or footpath visible (0=none, 1=dominant)
  natural_surfaces   - dirt, grass, gravel paths (0=none, 1=dominant)

Spatial qualities
  enclosure          - enclosed by buildings/walls (0=open, 1=enclosed)
  visibility         - sightlines (0=blocked, 1=long clear views)
  active_frontage    - shop windows, cafes, eyes-on-street (0=none, 1=active)
  blank_walls        - large featureless walls (0=none, 1=dominant)
  road_traffic       - cars or traffic visible (0=none, 1=heavy)

Overall impression
  perceived_safety   - your overall impression of how safe this space feels \
to a woman walking alone at night (0=very unsafe, 1=very safe)
  perceived_upkeep   - overall sense of care and maintenance (0=neglected, 1=well-kept)

RESPOND ONLY with a valid JSON object, no markdown, no explanation:
{"vividness":0.0,"brightness":0.0,"colour_warmth":0.0,"visual_complexity":0.0,\
"people_present":0.0,"activity_level":0.0,"greenery":0.0,"tree_canopy":0.0,\
"lighting_infrastructure":0.0,"lighting_quality":0.0,"cctv_visible":0.0,\
"benches_seating":0.0,"fencing_barriers":0.0,"maintenance_level":0.0,\
"graffiti":0.0,"garbage_litter":0.0,"unkempt_vegetation":0.0,\
"concrete_dominance":0.0,"brick_surfaces":0.0,"glass_facades":0.0,\
"asphalt_path":0.0,"natural_surfaces":0.0,"enclosure":0.0,"visibility":0.0,\
"active_frontage":0.0,"blank_walls":0.0,"road_traffic":0.0,\
"perceived_safety":0.0,"perceived_upkeep":0.0}
"""


# ---------------------------------------------------------------------------
# Load API keys
# ---------------------------------------------------------------------------
def load_keys():
    if not API_TXT.exists():
        sys.exit("[ERROR] api.txt not found")
    lines = API_TXT.read_text().splitlines()
    gemini_key = sv_key = None
    for i, line in enumerate(lines):
        tag = line.strip().lower()
        if "gemini" in tag:
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    gemini_key = lines[j].strip(); break
        if "google maps" in tag or "streetview" in tag or "street_view" in tag:
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    sv_key = lines[j].strip(); break
    if not gemini_key: sys.exit("[ERROR] Gemini key not found in api.txt")
    if not sv_key:     sys.exit("[ERROR] Street View key not found in api.txt")
    return gemini_key, sv_key


# ---------------------------------------------------------------------------
# Street View download
# ---------------------------------------------------------------------------
def sv_has_coverage(lat, lon, key):
    try:
        r = requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"location": f"{lat},{lon}", "key": key, "radius": 50},
            timeout=15,
        )
        return r.json().get("status") == "OK"
    except Exception:
        return False


def download_sv(lat, lon, heading, path: Path, key: str) -> bool:
    if path.exists() and path.stat().st_size > 2000:
        return True
    for attempt in range(3):
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/streetview",
                params={
                    "location": f"{lat},{lon}",
                    "size": SV_SIZE, "fov": SV_FOV,
                    "heading": heading, "pitch": SV_PITCH,
                    "key": key, "return_error_code": "true",
                },
                timeout=30,
            )
            if r.status_code != 200 or len(r.content) < 3000:
                return False
            path.write_bytes(r.content)
            return True
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                print(f"    [retry {attempt+1}] connection error, waiting 10s ...")
                time.sleep(10)
            else:
                return False
    return False


# ---------------------------------------------------------------------------
# Gemini Vision
# ---------------------------------------------------------------------------
def _cache_path(lat, lon):
    slug = f"{lat:.5f}_{lon:.5f}"
    h = hashlib.md5(slug.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def call_gemini(image_paths: list, key: str) -> dict | None:
    lat_str = str(image_paths[0]) if image_paths else ""
    cache = _cache_path(*_slug_from_paths(image_paths))
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            # Validate it has our keys
            if any(k in cached for k in SCORE_KEYS):
                return cached
        except Exception:
            pass

    parts = [{"text": ANALYSIS_PROMPT}]
    for p in image_paths:
        if Path(p).exists():
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    if len(parts) == 1:
        return None

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }

    for round_num in range(3):
        if round_num > 0:
            wait = 60 * round_num
            print(f"    [retry round {round_num}] waiting {wait}s ...")
            time.sleep(wait)

        for model in GEMINI_MODELS:
            cfg = payload["generationConfig"]
            if "2.5" in model and "lite" not in model:
                cfg["thinkingConfig"] = {"thinkingBudget": 0}
            elif "thinkingConfig" in cfg:
                del cfg["thinkingConfig"]

            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent")
            try:
                r = requests.post(
                    url, headers={"Content-Type": "application/json"},
                    params={"key": key}, json=payload, timeout=90,
                )
                if r.status_code in (429, 503):
                    print(f"    [{model}] rate limited, next model ...")
                    time.sleep(5)
                    continue
                if r.status_code >= 400:
                    print(f"    [{model}] HTTP {r.status_code}")
                    continue

                text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                if "```" in text:
                    text = text.rsplit("```", 1)[0]
                text = text.strip()
                if "{" in text:
                    start = text.index("{")
                    brace = 0
                    for ci, ch in enumerate(text[start:], start):
                        if ch == "{": brace += 1
                        elif ch == "}": brace -= 1
                        if brace == 0:
                            text = text[start:ci+1]; break

                scores = json.loads(text)
                for k in SCORE_KEYS:
                    if k in scores:
                        scores[k] = round(max(0.0, min(1.0, float(scores[k]))), 3)
                    else:
                        scores[k] = None

                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(scores, indent=2))
                return scores

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"    [{model}] parse error: {e}")
            except Exception as e:
                print(f"    [{model}] error: {e}")

    return None


def _slug_from_paths(paths):
    """Extract lat, lon from first path name like sv_{lat}_{lon}_N.jpg"""
    import re
    for p in paths:
        m = re.search(r"sv_([+-]?\d+\.\d+)_([+-]?\d+\.\d+)", str(p))
        if m:
            return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Step 1 — Street View download for all unique locations
# ---------------------------------------------------------------------------
def run_sv_download(locs_df, sv_key):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(locs_df)
    new_downloads = 0

    for i, row in locs_df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        slug = f"{lat:.5f}_{lon:.5f}"
        front = IMAGE_DIR / f"sv_{slug}_N.jpg"
        back  = IMAGE_DIR / f"sv_{slug}_S.jpg"

        already = front.exists() and front.stat().st_size > 2000

        if i % 50 == 0 or not already:
            print(f"  [{i+1}/{total}] ({lat:.5f}, {lon:.5f})", end="")

        if already:
            if i % 50 == 0:
                print(" [cached]")
            continue

        # Check SV coverage
        if not sv_has_coverage(lat, lon, sv_key):
            print(" [no coverage]")
            time.sleep(SV_DELAY)
            continue

        f_ok = download_sv(lat, lon, 0,   front, sv_key)
        time.sleep(SV_DELAY)
        b_ok = download_sv(lat, lon, 180, back,  sv_key)
        time.sleep(SV_DELAY)

        new_downloads += 1
        print(f" N={'ok' if f_ok else 'fail'} S={'ok' if b_ok else 'fail'}")

    print(f"\n[SV] Done. {new_downloads} new locations downloaded.")


# ---------------------------------------------------------------------------
# Step 2 — Gemini analysis for all unique locations
# ---------------------------------------------------------------------------
def run_gemini_analysis(locs_df, gemini_key):
    total = len(locs_df)
    results = []

    for i, row in locs_df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        slug = f"{lat:.5f}_{lon:.5f}"
        front = IMAGE_DIR / f"sv_{slug}_N.jpg"
        back  = IMAGE_DIR / f"sv_{slug}_S.jpg"
        imgs  = [p for p in [front, back] if p.exists()]

        # Check cache first (no print if cached)
        cache = _cache_path(lat, lon)
        cached = False
        if cache.exists():
            try:
                c = json.loads(cache.read_text())
                if any(k in c for k in SCORE_KEYS):
                    results.append({"latitude": lat, "longitude": lon, **c})
                    cached = True
            except Exception:
                pass

        if cached:
            if i % 100 == 0:
                print(f"  [{i+1}/{total}] cached")
            continue

        if not imgs:
            print(f"  [{i+1}/{total}] ({lat:.5f}, {lon:.5f}) no images")
            results.append({"latitude": lat, "longitude": lon,
                            **{k: None for k in SCORE_KEYS}})
            continue

        print(f"  [{i+1}/{total}] ({lat:.5f}, {lon:.5f}) {len(imgs)} images", end=" ")
        scores = call_gemini(imgs, gemini_key)
        if scores:
            print("ok")
            results.append({"latitude": lat, "longitude": lon, **scores})
        else:
            print("FAILED")
            results.append({"latitude": lat, "longitude": lon,
                            **{k: None for k in SCORE_KEYS}})

        time.sleep(GEMINI_DELAY)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 3 — Merge and build ML-ready CSV
# ---------------------------------------------------------------------------
def build_ml_csv(feat_df):
    df = pd.read_csv(YOURGROUND_CSV)

    # Engineer target columns
    def parse_stress(v):
        try:
            return int(float(v))
        except Exception:
            return None

    df["stress_num"]    = df["Stress_rating"].apply(parse_stress)
    df["stress_binary"] = df["stress_num"].apply(
        lambda x: 1 if x is not None and x >= 3 else (0 if x is not None else None)
    )
    df["is_after_dark"] = (df["Time Of Day"] == "After_dark").astype(int)
    df["is_dawn_dusk"]  = (df["Time Of Day"] == "Dawn_dusk").astype(int)
    df["is_daylight"]   = (df["Time Of Day"] == "Daylight").astype(int)
    df["has_good_light"]= (df["Lighting Is Good"] == "Lighting_is_good").astype(int)
    df["has_poor_light"]= (df["Poor Lighting"] == "Poor_lighting").astype(int)

    # Round for merge
    df["lat_r"]      = df["latitude"].round(4)
    df["lon_r"]      = df["longitude"].round(4)
    feat_df["lat_r"] = feat_df["latitude"].round(4)
    feat_df["lon_r"] = feat_df["longitude"].round(4)

    merged = df.merge(
        feat_df.drop(columns=["latitude","longitude"], errors="ignore"),
        on=["lat_r","lon_r"], how="left"
    ).drop(columns=["lat_r","lon_r"])

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[SAVED] {OUTPUT_CSV}  ({len(merged)} rows, {len(merged.columns)} columns)")
    return merged


# ---------------------------------------------------------------------------
# Step 4 — Correlation analysis report
# ---------------------------------------------------------------------------
def run_analysis(merged):
    lines = []
    lines.append("=" * 65)
    lines.append("LLM IMAGE ANALYSIS - CORRELATION WITH PERCEIVED SAFETY")
    lines.append("Full YourGround Dataset")
    lines.append("=" * 65)
    lines.append(f"Total reports    : {len(merged)}")
    scored = merged[SCORE_KEYS[0]].notna().sum()
    lines.append(f"Locations scored : {scored}")
    lines.append(f"Features scored  : {len(SCORE_KEYS)}")
    lines.append("")

    valid = merged.dropna(subset=["stress_num"])
    valid = valid[pd.to_numeric(valid["stress_num"], errors="coerce").notna()]
    valid["stress_num"] = pd.to_numeric(valid["stress_num"])

    # --- Correlation with stress ---
    lines.append("-" * 65)
    lines.append("CORRELATION WITH STRESS RATING (Pearson r, n=" + str(len(valid)) + ")")
    lines.append("Positive r -> higher score = higher stress (worse)")
    lines.append("Negative r -> higher score = lower stress (better)")
    lines.append("-" * 65)

    corrs = {}
    for k in SCORE_KEYS:
        col = pd.to_numeric(valid[k], errors="coerce")
        if col.notna().sum() > 10:
            corrs[k] = col.corr(valid["stress_num"])

    for feat, r in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "(+)" if r > 0 else "(-)"
        bar = "#" * int(abs(r) * 30)
        lines.append(f"  {feat:<28} r={r:+.3f}  {direction}  {bar}")

    # --- Mean by stress level ---
    lines.append("")
    lines.append("-" * 65)
    lines.append("MEAN VISUAL SCORES BY STRESS RATING (0-5)")
    lines.append("-" * 65)
    lines.append(f"{'Feature':<28}  " + "  ".join(f"  S{s}" for s in range(6)))
    for k in SCORE_KEYS:
        row_str = f"  {k:<28}  "
        for s in range(6):
            grp = merged[merged["stress_num"] == s]
            col  = pd.to_numeric(grp[k], errors="coerce")
            v    = col.mean()
            row_str += f"{v:5.2f}  " if not pd.isna(v) else "  --   "
        lines.append(row_str)

    # --- After dark vs daylight ---
    lines.append("")
    lines.append("-" * 65)
    lines.append("MEAN SCORES: AFTER-DARK vs DAYLIGHT REPORTS")
    lines.append("-" * 65)
    ad  = merged[merged["Time Of Day"] == "After_dark"]
    day = merged[merged["Time Of Day"] == "Daylight"]
    lines.append(f"n: after-dark={len(ad)}  daylight={len(day)}")
    lines.append(f"{'Feature':<28}  After-dark  Daylight    Diff")
    for k in SCORE_KEYS:
        a = pd.to_numeric(ad[k],  errors="coerce").mean()
        d = pd.to_numeric(day[k], errors="coerce").mean()
        if not (pd.isna(a) or pd.isna(d)):
            lines.append(f"  {k:<28}  {a:6.3f}      {d:6.3f}    {a-d:+.3f}")

    # --- Good lighting vs poor lighting reporters ---
    lines.append("")
    lines.append("-" * 65)
    lines.append("MEAN SCORES: POOR LIGHTING vs GOOD LIGHTING REPORTERS")
    lines.append("-" * 65)
    poor = merged[merged["Poor Lighting"] == "Poor_lighting"]
    good = merged[merged["Lighting Is Good"] == "Lighting_is_good"]
    lines.append(f"n: poor={len(poor)}  good={len(good)}")
    lines.append(f"{'Feature':<28}   Poor   Good   Diff")
    diffs = []
    for k in SCORE_KEYS:
        p = pd.to_numeric(poor[k], errors="coerce").mean()
        g = pd.to_numeric(good[k], errors="coerce").mean()
        if not (pd.isna(p) or pd.isna(g)):
            diffs.append((k, p, g, p - g))
    for feat, p, g, diff in sorted(diffs, key=lambda x: abs(x[3]), reverse=True):
        lines.append(f"  {feat:<28}  {p:.3f}  {g:.3f}  {diff:+.3f}")

    # --- ML-ready column summary ---
    lines.append("")
    lines.append("-" * 65)
    lines.append("ML-READY CSV COLUMNS")
    lines.append("-" * 65)
    lines.append("Target columns:")
    lines.append("  stress_num      (int 0-5, NaN if unspecified)")
    lines.append("  stress_binary   (1 = stress>=3, 0 = stress<3)")
    lines.append("  has_poor_light  (1 = Poor_lighting flag)")
    lines.append("  has_good_light  (1 = Lighting_is_good flag)")
    lines.append("Feature columns (visual, 0.0-1.0):")
    for k in SCORE_KEYS:
        lines.append(f"  {k}")
    lines.append("Context columns:")
    lines.append("  is_after_dark, is_dawn_dusk, is_daylight")
    lines.append("  latitude, longitude, LGA, Time Of Day, Stress_rating")

    report = "\n".join(lines)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n[SAVED] {OUTPUT_REPORT}")
    print()
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample",    type=int,            help="Limit to N unique locations")
    p.add_argument("--skip-sv",   action="store_true", help="Skip Street View download")
    p.add_argument("--skip-llm",  action="store_true", help="Skip Gemini, build CSV from cache")
    args = p.parse_args()

    gemini_key, sv_key = load_keys()
    print(f"[KEY] Gemini: {gemini_key[:12]}...  SV: {sv_key[:12]}...")

    # Unique locations
    df = pd.read_csv(YOURGROUND_CSV)
    df = df.dropna(subset=["latitude","longitude"])
    locs = df[["latitude","longitude"]].drop_duplicates().reset_index(drop=True)

    if args.sample:
        locs = locs.head(args.sample)

    print(f"[DATA] {len(locs)} unique locations to process")

    # Step 1 — Street View
    if not args.skip_sv:
        print(f"\n[SV] Downloading Street View images ...")
        run_sv_download(locs, sv_key)

    # Step 2 — Gemini
    if not args.skip_llm:
        print(f"\n[LLM] Running Gemini Vision analysis ...")
        feat_df = run_gemini_analysis(locs, gemini_key)
        feat_df.to_csv("_feat_df_checkpoint.csv", index=False)
    else:
        # Rebuild from cache
        print("[LLM] Rebuilding from cache ...")
        rows = []
        for _, row in locs.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            cache = _cache_path(lat, lon)
            if cache.exists():
                try:
                    c = json.loads(cache.read_text())
                    rows.append({"latitude": lat, "longitude": lon, **c})
                    continue
                except Exception:
                    pass
            rows.append({"latitude": lat, "longitude": lon,
                         **{k: None for k in SCORE_KEYS}})
        feat_df = pd.DataFrame(rows)

    # Step 3 — Merge
    print("\n[MERGE] Building ML-ready CSV ...")
    merged = build_ml_csv(feat_df)

    # Step 4 — Analysis
    print("\n[ANALYSIS] Computing correlations ...")
    run_analysis(merged)
