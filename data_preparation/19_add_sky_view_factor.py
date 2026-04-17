"""
19_add_sky_view_factor.py
=========================
Adds sky_view_factor to the Gemini cache for all locations that don't
already have it, then adds the column to yourground_ml_features.csv.

sky_view_factor: proportion of sky visible from street level (0=none, 1=full sky).
High sky = open, low sky = enclosed canyon. Related to building height/street
width ratio (H/W) but estimated directly from Street View images.

This is a targeted top-up — it does NOT re-run the full 29-feature analysis.
Each call sends a single focused question to Gemini, making it fast and cheap.

Usage:
    cd free_to_be_data
    python 19_add_sky_view_factor.py
"""

import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
CACHE_DIR  = Path("cache_llm_report")
IMAGE_DIR  = Path("report_images")
ML_CSV     = Path("yourground_ml_features.csv")
API_TXT    = Path("api.txt")

NEW_KEY    = "sky_view_factor"
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
DELAY      = 6   # seconds between Gemini calls

TOPUP_PROMPT = """\
You are an urban environment analyst. You will receive one or two street-level \
photographs of the same location (front-facing and back-facing views).

Score ONE feature only:

  sky_view_factor  - proportion of the sky visible from street level.
                     Consider the full hemisphere above the viewer.
                     0.0 = no sky visible (completely enclosed, deep canyon, tunnel)
                     0.5 = about half the sky visible (typical urban street with buildings)
                     1.0 = full open sky visible (open field, low-rise area, wide plaza)

RESPOND ONLY with a valid JSON object, no markdown, no explanation:
{"sky_view_factor": 0.0}
"""


# ---------------------------------------------------------------------------
def load_gemini_key():
    if not API_TXT.exists():
        sys.exit("[ERROR] api.txt not found")
    lines = API_TXT.read_text().splitlines()
    for i, line in enumerate(lines):
        if "gemini" in line.strip().lower():
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    return lines[j].strip()
    sys.exit("[ERROR] Gemini key not found in api.txt")


def cache_path(lat, lon):
    slug = f"{lat:.5f}_{lon:.5f}"
    h = hashlib.md5(slug.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def call_gemini_topup(image_paths, key):
    """Send targeted single-feature Gemini request."""
    parts = [{"text": TOPUP_PROMPT}]
    for p in image_paths:
        if Path(p).exists():
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    if len(parts) == 1:
        return None

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 64},
    }

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
                params={"key": key}, json=payload, timeout=60,
            )
            if r.status_code in (429, 503):
                time.sleep(10)
                continue
            if r.status_code >= 400:
                continue

            text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if "```" in text:
                text = text.rsplit("```", 1)[0]
            if "{" in text:
                start = text.index("{")
                brace = 0
                for ci, ch in enumerate(text[start:], start):
                    if ch == "{": brace += 1
                    elif ch == "}": brace -= 1
                    if brace == 0:
                        text = text[start:ci + 1]; break

            result = json.loads(text)
            val = result.get(NEW_KEY)
            if val is not None:
                return round(max(0.0, min(1.0, float(val))), 3)

        except Exception as e:
            print(f"    [{model}] error: {e}")
            continue

    return None


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("19_add_sky_view_factor.py")
    print("=" * 60)

    gemini_key = load_gemini_key()
    print(f"[KEY] Gemini: {gemini_key[:12]}...")

    df = pd.read_csv(ML_CSV)
    print(f"[DATA] {len(df):,} rows")

    # Get unique locations
    lat_col = "latitude" if "latitude" in df.columns else "Latitude"
    lon_col = "longitude" if "longitude" in df.columns else "Longitude"
    locs = df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    print(f"[DATA] {len(locs):,} unique locations")

    # Find which locations already have sky_view_factor in cache
    already_done = 0
    to_score = []
    for _, row in locs.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        cp = cache_path(lat, lon)
        if cp.exists():
            try:
                cached = json.loads(cp.read_text())
                if NEW_KEY in cached and cached[NEW_KEY] is not None:
                    already_done += 1
                    continue
            except Exception:
                pass
        to_score.append((lat, lon))

    print(f"[CACHE] {already_done} already have {NEW_KEY}")
    print(f"[TODO]  {len(to_score)} locations to score")

    if not to_score:
        print("[DONE] All locations already scored — rebuilding CSV column only")
    else:
        print(f"\nStarting Gemini top-up (~{len(to_score) * DELAY // 60} min estimated) ...")
        failed = 0
        for i, (lat, lon) in enumerate(to_score):
            slug = f"{lat:.5f}_{lon:.5f}"
            front = IMAGE_DIR / f"sv_{slug}_N.jpg"
            back  = IMAGE_DIR / f"sv_{slug}_S.jpg"
            imgs  = [p for p in [front, back] if p.exists()]

            print(f"  [{i+1}/{len(to_score)}] ({lat:.5f}, {lon:.5f})", end=" ")

            if not imgs:
                print("no images")
                failed += 1
                continue

            val = call_gemini_topup(imgs, gemini_key)

            if val is not None:
                # Update cache
                cp = cache_path(lat, lon)
                if cp.exists():
                    try:
                        cached = json.loads(cp.read_text())
                    except Exception:
                        cached = {}
                else:
                    cached = {}
                cached[NEW_KEY] = val
                cp.write_text(json.dumps(cached, indent=2))
                print(f"-> {val}")
            else:
                print("FAILED")
                failed += 1

            time.sleep(DELAY)

        print(f"\n[DONE] Scored {len(to_score) - failed}/{len(to_score)} locations")

    # Rebuild CSV column from cache
    print("\n[CSV] Adding sky_view_factor column to CSV ...")
    svf_map = {}
    for _, row in locs.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        cp = cache_path(lat, lon)
        if cp.exists():
            try:
                cached = json.loads(cp.read_text())
                svf_map[(round(lat, 5), round(lon, 5))] = cached.get(NEW_KEY)
            except Exception:
                pass

    df[NEW_KEY] = df.apply(
        lambda r: svf_map.get((round(r[lat_col], 5), round(r[lon_col], 5))),
        axis=1
    )

    coverage = df[NEW_KEY].notna().sum()
    print(f"[CSV] {coverage}/{len(df)} rows have {NEW_KEY} "
          f"({coverage/len(df)*100:.0f}%)")
    print(f"[CSV] Mean: {df[NEW_KEY].mean():.3f}  "
          f"Min: {df[NEW_KEY].min():.3f}  Max: {df[NEW_KEY].max():.3f}")

    df.to_csv(ML_CSV, index=False)
    print(f"[SAVED] {ML_CSV}")
