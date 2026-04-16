"""
regression.py
=============
Trains a Random Forest Regressor to predict the full perceived stress
score (0-5 gradient) rather than a binary safe/unsafe label.

This preserves the nuance of the original YourGround data — instead of
collapsing "slightly stressed" (2) and "very stressed" (5) into the same
class, the model learns to predict the actual degree of stress.

Features:
  - 29 Gemini Vision visual features (scored 0-1 from Street View images)
  - Space syntax measures (network connectivity, integration, mean depth)
  - OSM land use zone types
  - Environment type (park, street, trail, etc.)
  - Time-of-day flags

Outputs (saved to outputs/):
  report.txt                    -- R², MAE, RMSE and feature importances
  actual_vs_predicted.png       -- bar chart of mean predicted vs actual rating
  feature_importance.png        -- top 20 features
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data"
OUT_DIR    = BASE_DIR / "outputs"
ML_CSV     = DATA_DIR / "yourground_ml_features.csv"
OUT_REPORT = OUT_DIR / "report.txt"
OUT_PRED   = OUT_DIR / "actual_vs_predicted.png"
OUT_IMP    = OUT_DIR / "feature_importance.png"

OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
GEMINI_FEATURES = [
    "vividness", "brightness", "colour_warmth", "visual_complexity",
    "people_present", "activity_level", "greenery", "tree_canopy",
    "lighting_infrastructure", "lighting_quality", "cctv_visible",
    "benches_seating", "fencing_barriers", "maintenance_level", "graffiti",
    "garbage_litter", "unkempt_vegetation", "concrete_dominance",
    "brick_surfaces", "glass_facades", "asphalt_path", "natural_surfaces",
    "enclosure", "visibility", "active_frontage", "blank_walls",
    "road_traffic", "perceived_safety", "perceived_upkeep",
]

SS_FEATURES = [
    "ss_connectivity", "ss_integration", "ss_choice",
    "ss2_connectivity", "ss2_integration_local",
    "ss2_edge_length", "ss2_mean_depth",
]

CONTEXT_FEATURES = [
    "is_dawn_dusk", "is_daylight",
    "zone_residential", "zone_commercial", "zone_industrial",
    "zone_institutional", "zone_transport", "zone_parkland", "zone_unclassified",
]

CATEGORICAL_FEATURES = ["Environment", "Activity"]
TARGET = "stress_num"


# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(ML_CSV)
    df["stress_num"] = pd.to_numeric(df["stress_num"], errors="coerce")
    df = df.dropna(subset=[TARGET])

    all_features = GEMINI_FEATURES + SS_FEATURES + CONTEXT_FEATURES
    available = [f for f in all_features if f in df.columns]

    X = df[available].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in df.columns:
            continue
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col.lower(), dummy_na=False)
        dummies.columns = [c.replace(" ", "_").replace("/", "_") for c in dummies.columns]
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    y = df[TARGET].reset_index(drop=True)
    feature_names = list(X.columns)

    print(f"[DATA] {len(df):,} rows with stress rating (0-5)")
    print(f"[DATA] {len(feature_names)} features")
    print(f"[DATA] Stress distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y, feature_names


def train(X, y, feature_names):
    lines = []
    lines.append("=" * 70)
    lines.append("RANDOM FOREST REGRESSION — STRESS SCORE (0-5 gradient)")
    lines.append("=" * 70)
    lines.append(f"Features: {len(feature_names)}  |  n = {len(y):,}  |  Trees: 500")
    lines.append("")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("[REG]  Training Random Forest Regressor (500 trees) ...")
    reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(reg, X_train, y_train, cv=cv, scoring="r2")
    cv_mae = cross_val_score(reg, X_train, y_train, cv=cv,
                             scoring="neg_mean_absolute_error")

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    lines.append(f"  CV R²  : {cv_r2.mean():.4f}  (+/-{cv_r2.std():.4f})")
    lines.append(f"  CV MAE : {-cv_mae.mean():.4f}  (+/-{cv_mae.std():.4f})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("HOLD-OUT TEST SET (20%)")
    lines.append("-" * 70)
    lines.append(f"  R²   : {r2:.4f}  (variance in stress explained by the model)")
    lines.append(f"  MAE  : {mae:.4f}  (avg prediction error in stress units, scale 0-5)")
    lines.append(f"  RMSE : {rmse:.4f}")
    lines.append("")
    lines.append("  Interpretation:")
    lines.append(f"  The model explains {r2*100:.1f}% of variance in perceived stress scores.")
    lines.append(f"  On average, predictions are off by {mae:.2f} points on the 0-5 scale.")
    lines.append("  The remaining variance reflects personal, social, and contextual factors")
    lines.append("  not captured by the physical environment alone.")

    imp = pd.Series(reg.feature_importances_, index=feature_names).sort_values(ascending=False)
    lines.append("")
    lines.append("-" * 70)
    lines.append("FEATURE IMPORTANCES (top 25)")
    lines.append("-" * 70)
    for feat, val in imp.head(25).items():
        bar = "#" * int(val * 200)
        lines.append(f"  {feat:<32}  {val:.4f}  {bar}")

    return reg, imp, lines, y_test, y_pred


def save_outputs(imp, lines, y_test, y_pred):
    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"[SAVED] {OUT_REPORT}")

    # Actual vs predicted plot
    results = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
    means = results.groupby("actual")["predicted"].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(means.index, means.values, color="#1f77b4", alpha=0.8, label="Mean predicted")
    ax.plot([0, 5], [0, 5], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual stress rating (0-5)", fontsize=12)
    ax.set_ylabel("Mean predicted stress", fontsize=12)
    ax.set_title("Random Forest Regression: Actual vs Mean Predicted Stress", fontsize=13)
    ax.legend()
    ax.set_xticks(range(6))
    plt.tight_layout()
    fig.savefig(OUT_PRED, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {OUT_PRED}")

    # Feature importance plot
    top = imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    top.plot.barh(ax=ax, color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (mean decrease impurity)", fontsize=12)
    ax.set_title("Top 20 Features — RF Regression (Stress Score 0-5)", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_IMP, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {OUT_IMP}")

    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    X, y, feature_names = load_data()
    reg, imp, lines, y_test, y_pred = train(X, y, feature_names)
    save_outputs(imp, lines, y_test, y_pred)
    print("\nDone.")
