"""
17_add_land_use.py
==================
Fetches OSM land use / zoning polygons for the Melbourne metro area
and assigns zone type columns to each YourGround report location:

  zone_residential, zone_commercial, zone_industrial,
  zone_institutional, zone_transport, zone_parkland, zone_unclassified

Uses geopandas spatial join (point-in-polygon).
Adds columns to yourground_ml_features.csv.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox

ML_CSV      = Path("yourground_ml_features.csv")
CACHE_FILE  = Path("_melbourne_landuse.gpkg")

# OSM landuse tag values grouped into zone categories
ZONE_MAP = {
    "residential":   ["residential", "apartments"],
    "commercial":    ["commercial", "retail", "office", "business"],
    "industrial":    ["industrial", "warehouse", "construction"],
    "institutional": ["institutional", "education", "school", "university",
                      "college", "hospital", "religious", "civic",
                      "government", "public"],
    "transport":     ["transport", "railway", "aeroway", "parking",
                      "bus_station"],
    "parkland":      ["grass", "meadow", "recreation_ground", "park",
                      "nature_reserve", "forest", "wood", "conservation",
                      "cemetery", "allotments", "village_green",
                      "playground", "sports_centre", "stadium"],
}

BBOX = dict(north=-37.40, south=-38.52, east=145.85, west=144.35)


# ---------------------------------------------------------------------------
def load_landuse():
    if CACHE_FILE.exists():
        print(f"[LAND] Loading cached land use polygons from {CACHE_FILE} ...")
        gdf = gpd.read_file(CACHE_FILE)
        print(f"[LAND] {len(gdf):,} polygons loaded")
        return gdf

    print("[LAND] Downloading OSM land use polygons in tiles ...")
    ox.settings.timeout = 180
    ox.settings.max_query_area_size = 500_000_000

    tags = {"landuse": True, "leisure": True}

    # Split into 3x3 tiles to avoid Overpass query size limits
    lats = np.linspace(BBOX["south"], BBOX["north"], 4)
    lons = np.linspace(BBOX["west"],  BBOX["east"],  4)

    all_gdfs = []
    total_tiles = (len(lats) - 1) * (len(lons) - 1)
    tile_n = 0
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            tile_n += 1
            s, n = lats[i], lats[i + 1]
            w, e = lons[j], lons[j + 1]
            print(f"  Tile {tile_n}/{total_tiles}: "
                  f"lat [{s:.2f},{n:.2f}] lon [{w:.2f},{e:.2f}] ...")
            try:
                tile_gdf = ox.features_from_bbox(
                    bbox=(w, s, e, n), tags=tags
                )
                tile_gdf = tile_gdf[
                    tile_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                ].copy()
                keep = [c for c in ["landuse", "leisure", "geometry"]
                        if c in tile_gdf.columns]
                all_gdfs.append(tile_gdf[keep])
                print(f"    -> {len(tile_gdf):,} polygons")
            except Exception as e:
                print(f"    [WARN] Tile failed: {e}")

    if not all_gdfs:
        raise RuntimeError("All tiles failed — check internet connection")

    gdf = pd.concat(all_gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.drop_duplicates(subset="geometry").reset_index(drop=True)
    gdf.to_file(CACHE_FILE, driver="GPKG")
    print(f"[LAND] {len(gdf):,} polygons total, cached to {CACHE_FILE}")
    return gdf


def classify_zone(row):
    """Return zone category string for a landuse polygon row."""
    val = None
    for col in ["landuse", "leisure", "amenity"]:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).lower()
            break
    if val is None:
        return "unclassified"
    for zone, keywords in ZONE_MAP.items():
        if any(kw in val for kw in keywords):
            return zone
    return "unclassified"


def assign_zones(df, gdf):
    print("[JOIN] Assigning zone type to each report location ...")

    # Build GeoDataFrame of report points
    geometry = [Point(lon, lat) for lon, lat in
                zip(df["longitude"].values, df["latitude"].values)]
    reports_gdf = gpd.GeoDataFrame(df[["latitude", "longitude"]].copy(),
                                   geometry=geometry, crs="EPSG:4326")

    # Add zone column to land use gdf
    gdf = gdf.copy()
    gdf["zone"] = gdf.apply(classify_zone, axis=1)

    # Spatial join: which polygon does each point fall in?
    joined = gpd.sjoin(reports_gdf, gdf[["zone", "geometry"]],
                       how="left", predicate="within")

    # If a point falls in multiple polygons, keep first non-unclassified
    # group by original index, pick best zone
    joined = joined.reset_index()
    # 'index' column = original report row index
    def best_zone(grp):
        non_unc = grp[grp["zone"] != "unclassified"]
        if len(non_unc) > 0:
            return non_unc.iloc[0]["zone"]
        return grp.iloc[0]["zone"] if len(grp) > 0 else "unclassified"

    zone_series = (joined.groupby("index")
                         .apply(best_zone)
                         .reindex(range(len(df)))
                         .fillna("unclassified"))

    df = df.copy()
    df["zone_type"] = zone_series.values
    print(f"[JOIN] Zone distribution:\n{df['zone_type'].value_counts().to_string()}")

    # One-hot encode
    zones = ["residential", "commercial", "industrial",
             "institutional", "transport", "parkland", "unclassified"]
    for z in zones:
        df[f"zone_{z}"] = (df["zone_type"] == z).astype(int)

    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(ML_CSV)

    # Drop old zone columns if they exist
    drop = [c for c in df.columns if c.startswith("zone_")]
    if drop:
        df = df.drop(columns=drop)
        print(f"[DATA] Dropped {len(drop)} old zone columns")
    print(f"[DATA] {len(df):,} rows")

    gdf = load_landuse()
    df  = assign_zones(df, gdf)

    df.to_csv(ML_CSV, index=False)
    print(f"\n[SAVED] {ML_CSV}  — added zone_* columns")
    print("Zone columns:", [c for c in df.columns if c.startswith("zone_")])
