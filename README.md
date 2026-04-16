# YourGround Machine Learning

Machine learning models to predict **perceived safety** in Melbourne public spaces, using environmental features extracted from Google Street View images, space syntax network analysis, and OSM land use data.

This project is part of a thesis investigating women's perceived safety in urban environments using community-reported data from the [YourGround](https://yourground.com.au) platform.

---

## What this project does

For each of 3,420 community safety reports in Melbourne:

1. **Gemini Vision AI** analyses Google Street View images and scores 29 environmental features (greenery, road traffic, graffiti, lighting, enclosure, etc.) on a 0–1 scale
2. **Space syntax** measures how connected and reachable each location is within the street network (integration, mean depth, edge length, choice)
3. **OSM land use** assigns zone type (residential, commercial, parkland, transport, etc.) to each location
4. **Three ML models** predict perceived safety from these features

---

## Project structure

```
yourground-machine-learning/
│
├── data/
│   ├── yourground_ml_features.csv   ← main dataset (see data/README.md)
│   └── README.md                    ← how to obtain/generate the dataset
│
├── decision_tree/
│   ├── decision_tree.py             ← Decision Tree classifier
│   └── outputs/
│       ├── report.txt
│       ├── decision_tree.png
│       └── feature_importance.png
│
├── random_forest/
│   ├── random_forest.py             ← Random Forest classifier (500 trees)
│   └── outputs/
│       ├── report.txt
│       └── feature_importance.png
│
├── regression/
│   ├── regression.py                ← Random Forest Regressor (stress 0-5)
│   └── outputs/
│       ├── report.txt
│       ├── actual_vs_predicted.png
│       └── feature_importance.png
│
├── data_preparation/                ← scripts that built the dataset
│   ├── 14_full_dataset_analysis.py  ← Gemini Vision pipeline
│   ├── 15c_space_syntax_fast.py     ← Space syntax computation
│   └── 17_add_land_use.py           ← OSM land use assignment
│
└── requirements.txt
```

---

## Models

### 1. Decision Tree (`decision_tree/`)
- **Target**: `stress_binary` (0 = low stress, 1 = high stress)
- **Method**: Optimal depth selected via 5-fold cross-validation; SMOTE balancing on training set
- **Results**: Accuracy ~59%, ROC-AUC ~0.58, CV F1 ~0.62
- **Best for**: Interpretable rules — you can read exactly what the model learned

### 2. Random Forest (`random_forest/`)
- **Target**: `stress_binary` (0 = low stress, 1 = high stress)
- **Method**: 500 trees, SMOTE balancing, all features contribute
- **Results**: Accuracy ~63%, ROC-AUC ~0.62, CV F1 ~0.68
- **Best for**: Better accuracy; shows contribution of all 80 features including image features

### 3. Regression (`regression/`)
- **Target**: `stress_num` (continuous 0–5 stress score)
- **Method**: Random Forest Regressor, 500 trees, preserves full stress gradient
- **Results**: R² = 0.16, MAE = 1.32 (on a 0-5 scale)
- **Best for**: Understanding *how much* of stress is explained by the built environment

---

## Key findings

| Finding | Value |
|---|---|
| Physical environment explains | **16%** of stress variance (R²=0.16) |
| Classification accuracy | **63%** (Random Forest) |
| Top predictor | **Space syntax** — network isolation (`ss2_integration_local`, `ss2_mean_depth`) |
| Top image feature | **Road traffic** — visible cars increase stress |
| Safest zone type | **Parkland** — reduces predicted stress |
| Most stressful zone | **Transport** zones (stations, parking lots) |

The remaining 84% of variance reflects personal, social, and contextual factors not captured by the physical environment — which is itself an important finding about perceived safety.

---

## Feature groups

### Gemini Vision features (from Street View images)
Scored 0.0–1.0 by Gemini 2.5 Flash AI:

| Feature | Higher value means... |
|---|---|
| `road_traffic` | More cars/buses visible |
| `fencing_barriers` | More fences, walls, barriers |
| `greenery` | More grass, shrubs, plants |
| `tree_canopy` | More trees overhead |
| `enclosure` | More surrounded/hemmed-in feeling |
| `visibility` | Can see further ahead |
| `natural_surfaces` | More grass/dirt vs concrete |
| `lighting_infrastructure` | More streetlights visible |
| `graffiti` | More graffiti present |
| `maintenance_level` | Better maintained environment |
| `vividness` | More colourful scene |
| ... | (29 features total) |

### Space syntax features
Computed from OSM walking network (800m radius):

| Feature | Meaning |
|---|---|
| `ss2_integration_local` | How reachable this location is from nearby streets |
| `ss2_mean_depth` | Average distance to reach other nodes |
| `ss2_edge_length` | Length of street segments nearby |
| `ss_choice` | How often this location is on through-routes |

### Zone type features (OSM land use)
Binary flags: `zone_residential`, `zone_commercial`, `zone_industrial`, `zone_institutional`, `zone_transport`, `zone_parkland`, `zone_unclassified`

---

## Setup

### Requirements

- Python 3.10+
- See `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/papadosifouele/yourground-machine-learning.git
cd yourground-machine-learning

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Get the data

See [data/README.md](data/README.md) for instructions on obtaining `yourground_ml_features.csv`.

Place it at: `data/yourground_ml_features.csv`

---

## Running the models

Each model is fully self-contained. Run from the repo root:

```bash
# Decision Tree
python decision_tree/decision_tree.py

# Random Forest (classifier)
python random_forest/random_forest.py

# Regression (predicts stress 0-5)
python regression/regression.py
```

Outputs are saved to the `outputs/` folder inside each model directory.

---

## Data preparation pipeline

If you want to regenerate the dataset from scratch (requires API keys):

```bash
# Step 1 — Gemini Vision analysis of Street View images
# Requires: GEMINI_API_KEY and GOOGLE_MAPS_API_KEY
python data_preparation/14_full_dataset_analysis.py

# Step 2 — Space syntax (downloads Melbourne OSM network ~1.1GB)
python data_preparation/15c_space_syntax_fast.py

# Step 3 — OSM land use zones
python data_preparation/17_add_land_use.py
```

---

## Methodology notes

- **SMOTE balancing**: The YourGround dataset has ~60% high-stress reports. SMOTE (Synthetic Minority Oversampling Technique) creates synthetic low-stress samples in the training set only, ensuring the model doesn't simply predict "high stress" for everything. The test set uses the real distribution.

- **Excluded features**: Self-reported lighting flags (`has_good_light`, `has_poor_light`) and time-of-day (`is_after_dark`) were excluded from some model runs to avoid circular reasoning — they directly encode the user's safety perception rather than the physical environment.

- **Space syntax**: Measures were computed using an 800m ego-network around each report location. KDTree pre-filtering (~200 candidate nodes) replaced full-graph Dijkstra for a 100x speedup.

- **R² = 0.16**: This is not a weak result — it means the *built environment alone* explains 16% of perceived safety. The rest is personal context (who is reporting, their history, social conditions), which is consistent with urban safety literature.

---

## Citation

If you use this code or dataset in your research, please cite:

```
Papadosifouele [Year]. YourGround Machine Learning: Predicting Perceived Safety 
in Melbourne Public Spaces. GitHub: https://github.com/papadosifouele/yourground-machine-learning
```

---

## License

MIT License — see LICENSE file.
