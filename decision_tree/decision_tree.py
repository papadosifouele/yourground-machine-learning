"""
decision_tree.py
================
Trains a Decision Tree classifier to predict perceived safety
(stress_binary: 0 = low stress <3, 1 = high stress >=3) from:

  - 29 Gemini Vision visual features (greenery, road traffic, enclosure, etc.)
  - Space syntax measures (integration, mean depth, edge length, choice)
  - OSM land use zone types (residential, commercial, parkland, transport, etc.)
  - Environment type (park, street, trail, etc.) and Activity type
  - Time-of-day flags (daylight, dawn/dusk)

Dataset is SMOTE-balanced on the training split (minority class oversampled).
Best max_depth is selected via 5-fold cross-validation on F1 macro.

Outputs (saved to outputs/):
  report.txt          -- full metrics, feature importances, tree rules
  decision_tree.png   -- visual tree diagram
  feature_importance.png -- bar chart of top 20 features
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Paths — assumes yourground_ml_features.csv is in ../data/
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data"
OUT_DIR    = BASE_DIR / "outputs"
ML_CSV     = DATA_DIR / "yourground_ml_features.csv"
OUT_REPORT = OUT_DIR / "report.txt"
OUT_TREE   = OUT_DIR / "decision_tree.png"
OUT_IMP    = OUT_DIR / "feature_importance.png"

OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature definitions
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
TARGET = "stress_binary"


# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(ML_CSV)
    df["stress_binary"] = pd.to_numeric(df["stress_binary"], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

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

    print(f"[DATA] {len(df):,} rows, {len(feature_names)} features")
    print(f"[DATA] Class balance: low={int((y==0).sum()):,}  high={int((y==1).sum()):,}")
    return X, y, feature_names


def train(X, y, feature_names):
    lines = []
    lines.append("=" * 70)
    lines.append("DECISION TREE — PERCEIVED SAFETY (stress_binary)")
    lines.append("=" * 70)
    lines.append(f"Features: {len(feature_names)}  |  Rows: {len(y):,}")
    lines.append("")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    lines.append(f"SMOTE applied: training set balanced to 50/50")
    lines.append("")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lines.append("-" * 70)
    lines.append("DEPTH SEARCH (5-fold CV, F1 macro)")
    lines.append("-" * 70)
    best_depth, best_f1 = 3, 0.0
    for depth in [3, 4, 5, 6, 8, 10, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
        label = str(depth) if depth else "None(full)"
        lines.append(f"  max_depth={label:<10}  CV F1={scores.mean():.4f}  (+/-{scores.std():.4f})")
        if scores.mean() > best_f1:
            best_f1, best_depth = scores.mean(), depth
    lines.append(f"\n  -> Best: max_depth={best_depth}  CV F1={best_f1:.4f}")

    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"HOLD-OUT TEST SET (20%, n={len(y_test):,})")
    lines.append("-" * 70)
    lines.append(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    lines.append(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    lines.append("")
    report = classification_report(y_test, y_pred, target_names=["Low stress", "High stress"])
    for l in report.splitlines():
        lines.append("    " + l)

    lines.append("")
    cm = confusion_matrix(y_test, y_pred)
    lines.append("  Confusion matrix (rows=actual, cols=predicted):")
    lines.append(f"    {'':15}  Pred Low  Pred High")
    lines.append(f"    {'Actual Low':<15}  {cm[0,0]:8d}  {cm[0,1]:9d}")
    lines.append(f"    {'Actual High':<15}  {cm[1,0]:8d}  {cm[1,1]:9d}")

    imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    lines.append("")
    lines.append("-" * 70)
    lines.append("FEATURE IMPORTANCES (Gini)")
    lines.append("-" * 70)
    for feat, val in imp[imp > 0].items():
        bar = "#" * int(val * 60)
        lines.append(f"  {feat:<32}  {val:.4f}  {bar}")

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"DECISION TREE RULES (max_depth={best_depth})")
    lines.append("-" * 70)
    rules = export_text(clf, feature_names=list(feature_names), max_depth=6)
    for l in rules.splitlines():
        lines.append("  " + l)

    return clf, imp, lines, X_test, y_test, feature_names


def save_outputs(clf, imp, lines, feature_names):
    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"[SAVED] {OUT_REPORT}")

    # Tree plot
    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(clf, feature_names=list(feature_names),
              class_names=["Low stress", "High stress"],
              filled=True, rounded=True, fontsize=8, max_depth=4, ax=ax)
    plt.title(f"Decision Tree — Perceived Safety (shown to depth 4)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_TREE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {OUT_TREE}")

    # Importance plot
    top = imp[imp > 0].head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    top.plot.barh(ax=ax, color="#1f77b4")
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (Gini)", fontsize=12)
    ax.set_title("Top 20 Features — Decision Tree (Perceived Safety)", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_IMP, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {OUT_IMP}")

    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    X, y, feature_names = load_data()
    clf, imp, lines, X_test, y_test, feat_names = train(X, y, feature_names)
    save_outputs(clf, imp, lines, feat_names)
    print("\nDone.")
