"""
15c_space_syntax_fast.py
=========================
Fast, correct space syntax measures for all YourGround report locations.

Works directly on the primal OSM graph (nodes = intersections, edges = streets).
For each report location:
  1. KDTree bounding-box query to find all nodes within 800m radius (~200 nodes)
  2. Extract tiny subgraph of just those nodes
  3. Run Dijkstra + betweenness on the small subgraph (milliseconds)

  ss_integration  – closeness centrality, length-weighted, 800m subgraph
  ss_choice       – betweenness centrality on local subgraph
  ss_connectivity – degree (instant)
  ss_edge_length  – mean length of edges incident to nearest node
  ss_mean_depth   – mean weighted distance to all reachable nodes in 800m

All five columns added / updated in yourground_ml_features.csv.

Runtime: ~5-10 minutes for 3,420 locations.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from scipy.spatial import KDTree

ML_CSV      = Path("yourground_ml_features.csv")
GRAPH_CACHE = Path("_melbourne_graph.graphml")
OUT_REPORT  = Path("space_syntax_angular_report.txt")

LOCAL_RADIUS_M = 800
M_LAT = 111_000
M_LON = 111_000 * math.cos(math.radians(-37.82))
# Bounding box radius in degrees (slight overestimate, then exact filter below)
BBOX_R_LAT = (LOCAL_RADIUS_M / M_LAT) * 1.05
BBOX_R_LON = (LOCAL_RADIUS_M / M_LON) * 1.05


# ---------------------------------------------------------------------------
def load_graph():
    print("[GRAPH] Loading cached primal graph ...")
    G = ox.load_graphml(GRAPH_CACHE)
    G_un = G.to_undirected()
    print(f"[GRAPH] {G_un.number_of_nodes():,} nodes  {G_un.number_of_edges():,} edges")
    return G_un


# ---------------------------------------------------------------------------
def compute_for_reports(G, report_lats, report_lons):
    # Build node coordinate arrays once
    node_ids = list(G.nodes())
    node_arr = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in node_ids])
    tree = KDTree(node_arr)

    print(f"[SS]   Computing for {len(report_lats):,} report locations ...")

    ss_connectivity = []
    ss_integration  = []
    ss_choice       = []
    ss_mean_depth   = []
    ss_edge_length  = []

    total = len(report_lats)

    for i, (lat, lon) in enumerate(zip(report_lats, report_lons)):
        if i % 200 == 0:
            print(f"  [{i}/{total}]", flush=True)

        # --- Nearest node ---
        _, nidx = tree.query([lat, lon], k=1)
        nid = node_ids[nidx]

        # Degree (connectivity) — uses full graph so all edges counted
        ss_connectivity.append(G.degree(nid))

        # Mean incident edge length
        inc_lengths = [d.get("length", 0) for _, _, d in G.edges(nid, data=True)]
        ss_edge_length.append(float(np.mean(inc_lengths)) if inc_lengths else 0.0)

        # --- Pre-filter to bounding box (fast KDTree) ---
        # This gives us ~100-300 candidate nodes instead of 847k
        bbox_idxs = tree.query_ball_point([lat, lon],
                                          r=max(BBOX_R_LAT, BBOX_R_LON))
        # Exact distance filter
        local_nodes = []
        for idx in bbox_idxs:
            dlat = (node_arr[idx, 0] - lat) * M_LAT
            dlon = (node_arr[idx, 1] - lon) * M_LON
            if math.sqrt(dlat * dlat + dlon * dlon) <= LOCAL_RADIUS_M:
                local_nodes.append(node_ids[idx])

        if len(local_nodes) < 3 or nid not in local_nodes:
            ss_integration.append(0.0)
            ss_choice.append(0.0)
            ss_mean_depth.append(0.0)
            continue

        # --- Build tiny subgraph (milliseconds) ---
        sub = G.subgraph(local_nodes)

        # Integration: closeness centrality of nid in local subgraph
        try:
            cl = nx.closeness_centrality(sub, u=nid, distance="length")
            ss_integration.append(cl if cl is not None else 0.0)
        except Exception:
            ss_integration.append(0.0)

        # Mean depth: mean shortest path length from nid to all nodes in subgraph
        try:
            dists = nx.single_source_dijkstra_path_length(sub, nid, weight="length")
            vals = [d for n, d in dists.items() if n != nid]
            ss_mean_depth.append(float(np.mean(vals)) if vals else 0.0)
        except Exception:
            ss_mean_depth.append(0.0)

        # Choice: approximate betweenness using k=15 random sources (fast)
        try:
            n_local = len(local_nodes)
            k = min(15, n_local)
            bc = nx.betweenness_centrality(sub, k=k, normalized=True, weight="length", seed=42)
            ss_choice.append(bc.get(nid, 0.0))
        except Exception:
            ss_choice.append(0.0)

    return ss_connectivity, ss_integration, ss_choice, ss_mean_depth, ss_edge_length


# ---------------------------------------------------------------------------
def analyse_and_save(df):
    cols   = ["ss_connectivity","ss_integration","ss_choice",
              "ss_mean_depth","ss_edge_length"]
    labels = ["Connectivity","Integration (local)","Choice (local)",
              "Mean depth","Edge length"]

    df["stress_num"] = pd.to_numeric(df["stress_num"], errors="coerce")
    valid = df.dropna(subset=["stress_num"])

    lines = []
    lines.append("=" * 65)
    lines.append("SPACE SYNTAX — CORRELATION WITH PERCEIVED SAFETY")
    lines.append("Primal graph, length-weighted, 800m ego-network")
    lines.append("=" * 65)
    lines.append(f"n = {len(valid):,} reports with stress rating\n")

    # Descriptive stats
    lines.append("-" * 65)
    lines.append("DESCRIPTIVE STATISTICS (raw values)")
    lines.append("-" * 65)
    for col, label in zip(cols, labels):
        s = df[col].dropna()
        lines.append(f"  {label:<22}  mean={s.mean():.4f}  "
                     f"std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")

    # Correlation with stress
    lines.append("")
    lines.append("-" * 65)
    lines.append("CORRELATION WITH STRESS RATING (Pearson r)")
    lines.append("(+) = higher value -> higher stress")
    lines.append("(-) = higher value -> lower stress")
    lines.append("-" * 65)
    for col, label in zip(cols, labels):
        r = pd.to_numeric(valid[col], errors="coerce").corr(valid["stress_num"])
        if pd.isna(r):
            lines.append(f"  {label:<22}  r=NaN"); continue
        bar  = "#" * int(abs(r) * 40)
        sign = "(+)" if r > 0 else "(-)"
        lines.append(f"  {label:<22}  r={r:+.4f}  {sign}  {bar}")

    # High vs low stress
    lines.append("")
    lines.append("-" * 65)
    lines.append("HIGH STRESS (>=3) vs LOW STRESS (<3)")
    lines.append("-" * 65)
    hi = df[df["stress_binary"] == 1]
    lo = df[df["stress_binary"] == 0]
    lines.append(f"  n: high={len(hi):,}  low={len(lo):,}")
    lines.append(f"  {'Measure':<22}    High      Low      Diff")
    for col, label in zip(cols, labels):
        h = pd.to_numeric(hi[col], errors="coerce").mean()
        l = pd.to_numeric(lo[col], errors="coerce").mean()
        lines.append(f"  {label:<22}  {h:8.4f}  {l:8.4f}  {h-l:+.4f}")

    # After dark vs daylight
    lines.append("")
    lines.append("-" * 65)
    lines.append("AFTER-DARK vs DAYLIGHT")
    lines.append("-" * 65)
    ad  = df[df["Time Of Day"] == "After_dark"]
    day = df[df["Time Of Day"] == "Daylight"]
    lines.append(f"  n: after-dark={len(ad):,}  daylight={len(day):,}")
    lines.append(f"  {'Measure':<22}  After-dark  Daylight    Diff")
    for col, label in zip(cols, labels):
        a = pd.to_numeric(ad[col],  errors="coerce").mean()
        d = pd.to_numeric(day[col], errors="coerce").mean()
        lines.append(f"  {label:<22}  {a:8.4f}    {d:8.4f}  {a-d:+.4f}")

    # Poor vs good lighting
    lines.append("")
    lines.append("-" * 65)
    lines.append("POOR LIGHTING vs GOOD LIGHTING REPORTERS")
    lines.append("-" * 65)
    poor = df[df["Poor Lighting"] == "Poor_lighting"]
    good = df[df["Lighting Is Good"] == "Lighting_is_good"]
    lines.append(f"  n: poor={len(poor):,}  good={len(good):,}")
    lines.append(f"  {'Measure':<22}     Poor      Good     Diff")
    for col, label in zip(cols, labels):
        p = pd.to_numeric(poor[col], errors="coerce").mean()
        g = pd.to_numeric(good[col], errors="coerce").mean()
        lines.append(f"  {label:<22}  {p:8.4f}  {g:8.4f}  {p-g:+.4f}")

    # Mean by stress level
    lines.append("")
    lines.append("-" * 65)
    lines.append("MEAN VALUES BY STRESS RATING (0-5)")
    lines.append("-" * 65)
    lines.append(f"  {'Measure':<22}" + "".join(f"      S{s}" for s in range(6)))
    for col, label in zip(cols, labels):
        row = f"  {label:<22}"
        for s in range(6):
            v = pd.to_numeric(df[df["stress_num"] == s][col],
                              errors="coerce").mean()
            row += f"  {v:6.3f}" if not pd.isna(v) else "      --"
        lines.append(row)

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n[SAVED] {OUT_REPORT}")
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(ML_CSV)
    # Drop old ss columns
    drop = [c for c in df.columns if c.startswith("ss_") or c.startswith("ss2_")]
    if drop:
        df = df.drop(columns=drop)
        print(f"[DATA] Dropped {len(drop)} old ss columns")
    df["stress_num"] = pd.to_numeric(df["stress_num"], errors="coerce")
    print(f"[DATA] {len(df):,} rows")

    G = load_graph()

    conn, integ, choice, depth, elen = compute_for_reports(
        G, df["latitude"].values, df["longitude"].values
    )

    df["ss_connectivity"] = conn
    df["ss_integration"]  = integ
    df["ss_choice"]       = choice
    df["ss_mean_depth"]   = depth
    df["ss_edge_length"]  = elen

    # Normalise 0-1
    for col in ["ss_connectivity","ss_integration","ss_choice",
                "ss_mean_depth","ss_edge_length"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_norm"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0

    df.to_csv(ML_CSV, index=False)
    print(f"[SAVED] {ML_CSV}")

    analyse_and_save(df)
