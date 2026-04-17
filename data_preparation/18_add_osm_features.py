"""
18_add_osm_features.py
======================
Downloads OSM features for Melbourne and adds four new feature groups
to yourground_ml_features.csv:

  Eyes on the street — POI density (200m)
    poi_eyes_200m        -- active-frontage POI: cafes, restaurants, bars,
                           shops, markets (Jane Jacobs "eyes on the street")
    poi_services_200m    -- services: banks, pharmacy, post office, library
    poi_density_200m     -- total POI count within 200m

  Transit accessibility (500m)
    transit_nearest_m    -- distance in metres to nearest stop/station
    transit_count_500m   -- count of stops/stations within 500m

  Pedestrian infrastructure (100m)
    has_footpath         -- dedicated footway/path within 100m (0/1)
    has_cycleway         -- cycling infrastructure within 100m (0/1)
    pedestrian_infra_score -- sum of above (0-2)

  Building density (100m)
    building_count_100m  -- number of building footprints within 100m
    building_density_100m -- buildings per hectare

All data from OSM — free, no API key, works for any city globally.

Usage:
    cd free_to_be_data
    python 18_add_osm_features.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point
import osmnx as ox

# ---------------------------------------------------------------------------
ML_CSV           = Path("yourground_ml_features.csv")
CACHE_POI        = Path("_cache_poi.gpkg")
CACHE_TRANSIT    = Path("_cache_transit.gpkg")
CACHE_PEDINFRA   = Path("_cache_pedinfra.gpkg")
CACHE_BUILDINGS  = Path("_cache_buildings.gpkg")

BBOX = {"north": -37.40, "south": -38.52, "east": 145.85, "west": 144.35}

# Melbourne local CRS in metres (GDA94 / MGA zone 55)
CRS_M = "EPSG:28355"

ox.settings.timeout = 180
ox.settings.max_query_area_size = 500_000_000

# POI tag classification
EYES_ON_STREET_AMENITIES = [
    "cafe", "restaurant", "bar", "pub", "fast_food", "food_court",
    "ice_cream", "biergarten",          # food & drink — highest foot traffic
    "market", "marketplace",            # outdoor markets
]
SERVICE_AMENITIES = [
    "bank", "post_office", "pharmacy", "clinic", "doctors",
    "library", "community_centre", "place_of_worship",
]

NEW_COLS = [
    "poi_eyes_200m", "poi_services_200m", "poi_density_200m",
    "transit_nearest_m", "transit_count_500m",
    "has_footpath", "has_cycleway", "pedestrian_infra_score",
    "building_count_100m", "building_density_100m",
]


# ---------------------------------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------------------------------

def tiled_download(tags, cache_file, geom_types, keep_cols, label):
    """Download OSM features in a 3x3 tile grid, cache to gpkg."""
    if cache_file.exists():
        print(f"[OSM] Loading cached {label} from {cache_file} ...")
        gdf = gpd.read_file(cache_file)
        print(f"[OSM] {len(gdf):,} {label} loaded")
        return gdf

    print(f"[OSM] Downloading {label} in tiles ...")
    lats = np.linspace(BBOX["south"], BBOX["north"], 4)
    lons = np.linspace(BBOX["west"],  BBOX["east"],  4)
    all_gdfs = []
    total = (len(lats) - 1) * (len(lons) - 1)
    tile_n = 0
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            tile_n += 1
            s, n = lats[i], lats[i + 1]
            w, e = lons[j], lons[j + 1]
            print(f"  Tile {tile_n}/{total}: "
                  f"lat[{s:.2f},{n:.2f}] lon[{w:.2f},{e:.2f}] ...")
            try:
                tile = ox.features_from_bbox(bbox=(w, s, e, n), tags=tags)
                tile = tile[tile.geometry.geom_type.isin(geom_types)].copy()
                cols = ["geometry"] + [c for c in keep_cols if c in tile.columns]
                all_gdfs.append(tile[cols])
                print(f"    -> {len(tile):,} features")
            except Exception as ex:  # noqa: BLE001
                print(f"    [WARN] Tile failed: {ex}")

    if not all_gdfs:
        raise RuntimeError(f"All tiles failed for {label}")

    result = pd.concat(all_gdfs, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")
    result = result.drop_duplicates(subset="geometry").reset_index(drop=True)
    result.to_file(cache_file, driver="GPKG")
    print(f"[OSM] {len(result):,} {label} cached to {cache_file}")
    return result


def load_poi():
    tags = {
        "amenity": EYES_ON_STREET_AMENITIES + SERVICE_AMENITIES,
        "shop": True,
    }
    return tiled_download(
        tags, CACHE_POI,
        geom_types=["Point"],
        keep_cols=["amenity", "shop"],
        label="POI (cafes, shops, services)",
    )


def load_transit():
    tags = {
        "highway": "bus_stop",
        "railway": ["station", "tram_stop", "halt", "stop"],
        "public_transport": ["stop_position", "station"],
    }
    return tiled_download(
        tags, CACHE_TRANSIT,
        geom_types=["Point"],
        keep_cols=[],
        label="transit stops",
    )


def load_ped_infra():
    tags = {
        "highway": ["footway", "path", "pedestrian", "cycleway", "steps"],
        "cycleway": True,
    }
    return tiled_download(
        tags, CACHE_PEDINFRA,
        geom_types=["LineString", "MultiLineString"],
        keep_cols=["highway", "cycleway"],
        label="pedestrian/cycling infrastructure",
    )


def load_buildings():
    tags = {"building": True}
    return tiled_download(
        tags, CACHE_BUILDINGS,
        geom_types=["Polygon", "MultiPolygon"],
        keep_cols=[],
        label="building footprints",
    )


# ---------------------------------------------------------------------------
# SPATIAL HELPERS
# ---------------------------------------------------------------------------

def to_projected(gdf_latlon):
    """Project a GeoDataFrame to Melbourne metres CRS."""
    return gdf_latlon.to_crs(CRS_M)


def make_report_gdf(df):
    """Build a projected GeoDataFrame from the reports CSV."""
    lat_col = "latitude" if "latitude" in df.columns else "Latitude"
    lon_col = "longitude" if "longitude" in df.columns else "Longitude"
    geom = [Point(lon, lat)
            for lon, lat in zip(df[lon_col].values, df[lat_col].values)]
    gdf = gpd.GeoDataFrame(
        {"_idx": range(len(df))}, geometry=geom, crs="EPSG:4326"
    )
    return to_projected(gdf)


def get_centroids_m(feat_gdf):
    """Return (N,2) array of projected centroids for any geometry type."""
    proj = to_projected(feat_gdf.copy())
    return np.array([[g.centroid.x, g.centroid.y] for g in proj.geometry])


def count_in_buffer(rep_coords, feat_coords, radius_m, label):
    """Count feat points within radius_m of each report. Returns array."""
    print(f"[COUNT] {label} within {radius_m}m ...")
    tree = KDTree(feat_coords)
    counts = np.array([len(tree.query_ball_point(pt, r=radius_m))
                       for pt in rep_coords])
    print(f"[COUNT] Done — mean {counts.mean():.1f}, max {counts.max()}")
    return counts


def nearest_m(rep_coords, feat_coords, label):
    """Distance in metres to nearest feature for each report."""
    print(f"[DIST] Nearest {label} ...")
    tree = KDTree(feat_coords)
    dists, _ = tree.query(rep_coords, k=1)
    print(f"[DIST] Median nearest: {np.median(dists):.0f}m")
    return dists


# ---------------------------------------------------------------------------
# FEATURE FUNCTIONS
# ---------------------------------------------------------------------------

def add_poi_features(df, rep_coords, poi_gdf):
    print("\n--- POI / Eyes on the street ---")
    poi_m = get_centroids_m(poi_gdf)

    # Eyes on the street: food/drink + shops
    if "amenity" in poi_gdf.columns:
        eyes_mask = poi_gdf["amenity"].isin(EYES_ON_STREET_AMENITIES) | \
                    poi_gdf["shop"].notna()
        svc_mask  = poi_gdf["amenity"].isin(SERVICE_AMENITIES)
    else:
        # Fallback if tag columns weren't saved (cache from old run)
        eyes_mask = pd.Series([True] * len(poi_gdf))
        svc_mask  = pd.Series([False] * len(poi_gdf))

    eyes_coords = get_centroids_m(poi_gdf[eyes_mask])
    svc_coords  = get_centroids_m(poi_gdf[svc_mask])

    df["poi_eyes_200m"]     = count_in_buffer(rep_coords, eyes_coords,
                                               200, "eyes-on-street POI")
    df["poi_services_200m"] = count_in_buffer(rep_coords, svc_coords,
                                               200, "service POI")
    df["poi_density_200m"]  = count_in_buffer(rep_coords, poi_m,
                                               200, "all POI")
    return df


def add_transit_features(df, rep_coords, transit_gdf):
    print("\n--- Transit accessibility ---")
    transit_m = get_centroids_m(transit_gdf)
    df["transit_nearest_m"]  = nearest_m(rep_coords, transit_m, "transit stop")
    df["transit_count_500m"] = count_in_buffer(rep_coords, transit_m,
                                                500, "transit stops")
    return df


def add_ped_infra_features(df, rep_coords, ped_gdf):
    print("\n--- Pedestrian infrastructure ---")
    ped_m = get_centroids_m(ped_gdf)

    # Split footpaths vs cycleways if tag column available
    if "highway" in ped_gdf.columns and "cycleway" in ped_gdf.columns:
        cycle_mask = (ped_gdf["highway"] == "cycleway") | \
                     (ped_gdf["cycleway"].notna())
        foot_mask  = ~cycle_mask
        foot_coords  = get_centroids_m(ped_gdf[foot_mask])
        cycle_coords = get_centroids_m(ped_gdf[cycle_mask])
    else:
        foot_coords  = ped_m
        cycle_coords = ped_m

    df["has_footpath"] = (count_in_buffer(rep_coords, foot_coords,
                                           100, "footpaths") > 0).astype(int)
    df["has_cycleway"] = (count_in_buffer(rep_coords, cycle_coords,
                                           100, "cycleways") > 0).astype(int)
    df["pedestrian_infra_score"] = df["has_footpath"] + df["has_cycleway"]
    return df


def add_building_features(df, rep_coords, buildings_gdf):
    print("\n--- Building density ---")
    bldg_m = get_centroids_m(buildings_gdf)
    counts = count_in_buffer(rep_coords, bldg_m, 100, "buildings")
    area_ha = math.pi * (100 ** 2) / 10_000   # 100m radius circle in ha
    df["building_count_100m"]   = counts
    df["building_density_100m"] = (counts / area_ha).round(2)
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("18_add_osm_features.py")
    print("=" * 60)

    df = pd.read_csv(ML_CSV)
    print(f"[DATA] {len(df):,} rows loaded")

    drop = [c for c in NEW_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        print(f"[DATA] Dropped {len(drop)} old columns for re-run")

    reports_gdf = make_report_gdf(df)
    rep_coords  = np.array([[g.x, g.y] for g in reports_gdf.geometry])

    poi_gdf       = load_poi()
    transit_gdf   = load_transit()
    ped_gdf       = load_ped_infra()
    buildings_gdf = load_buildings()

    df = add_poi_features(df, rep_coords, poi_gdf)
    df = add_transit_features(df, rep_coords, transit_gdf)
    df = add_ped_infra_features(df, rep_coords, ped_gdf)
    df = add_building_features(df, rep_coords, buildings_gdf)

    df.to_csv(ML_CSV, index=False)
    added = [c for c in NEW_COLS if c in df.columns]
    print(f"\n[SAVED] {ML_CSV}")
    print(f"[DONE] {len(added)} new columns: {added}")
    print("\nSample statistics:")
    print(df[added].describe().round(2).to_string())
