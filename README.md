# 🏙️ City Energy Consumption Analysis & Prediction System

A complete end-to-end data science project that simulates, analyzes, and forecasts daily electricity consumption across 5 city zones using machine learning.

---

## 📁 Project Structure

```
city_energy_system/
├── energy_system.py           # Core module (data gen, analysis, ML, CLI)
├── city_energy_analysis.ipynb # Jupyter Notebook (full walkthrough)
├── outputs/
│   ├── city_energy_dataset.csv        # Generated dataset
│   ├── energy_model.pkl               # Trained Random Forest model
│   ├── plot_monthly_trends.png        # Line chart – monthly trends
│   ├── plot_correlation_heatmap.png   # Feature correlation heatmap
│   ├── plot_event_vs_nonevent.png     # Bar chart – event vs normal days
│   ├── plot_zone_distribution.png     # Violin plot – consumption distributions
│   └── plot_feature_importance.png   # RF feature importances
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Run the full pipeline (CLI + interactive predictor)

```bash
python energy_system.py
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook city_energy_analysis.ipynb
```

Open `city_energy_analysis.ipynb` in your browser and run all cells from top to bottom.

---

## 📊 Dataset

The dataset is **synthetically generated** using NumPy and Pandas to simulate one year of daily electricity readings across 5 city zones.

### Zone Profiles

| Zone ID | Profile | Base Load | Key Driver |
|---|---|---|---|
| `Z1_Downtown` | Commercial / Office | ~5,800 kWh | Temperature (HVAC) |
| `Z2_Residential` | Residential North | ~3,200 kWh | Season + Events |
| `Z3_Industrial` | Industrial West | ~9,500 kWh | Always-on + Weekend shutdowns |
| `Z4_Suburban` | Suburban East | ~4,100 kWh | Humidity discomfort |
| `Z5_Entertainment` | Entertainment District | ~3,900 kWh | Events (2× frequency) |

### Features

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Day of reading |
| `ZoneID` | string | City zone identifier |
| `DayOfWeek` | int | 0=Monday … 6=Sunday |
| `Month` | int | 1–12 |
| `IsWeekend` | int | 1 if Saturday/Sunday |
| `AvgTemperature` | float | Daily average °C |
| `Humidity` | float | Percentage humidity |
| `SpecialEvent` | int | 0 = normal, 1 = event |
| `EnergyConsumption` | float | Total kWh for the zone-day |

---

## 🔍 Key Insights from Analysis

1. **Seasonal U-curve**: All zones peak in winter (heating) and summer (cooling). Spring/autumn are the most energy-efficient periods.

2. **Event impact varies by zone**:
   - Entertainment District: **+21.3%** on event days
   - Residential: **+12.0%**
   - Industrial: **−0.4%** (factories close for public holidays)

3. **Industrial weekend drop**: Factory closures reduce Industrial zone consumption by ~25% on weekends — the most dramatic behavioral pattern in the dataset.

4. **Rolling history is the strongest predictor**: The 7-day rolling average explains ~81% of the Random Forest's decision making. This reflects inertia in energy demand — tomorrow's demand is closely anchored to recent patterns.

5. **Humidity matters more in suburban zones**: Z4 Suburban shows a higher humidity coefficient, likely reflecting higher residential density and less commercial air-conditioning efficiency.

---

## 🤖 Machine Learning Model

### Algorithm: Random Forest Regressor

Chosen over Linear Regression because:
- Captures **non-linear** temperature effects (V-shaped heating + cooling curve)
- Handles **interaction effects** between event days and zone type
- More robust to outliers in weather data

### Feature Engineering

Beyond the raw features, the model uses:
- `Lag1_Consumption` – previous day's actual consumption
- `Rolling7_Consumption` – 7-day rolling average

### Results

| Model | MAE Train | MAE Test |
|---|---|---|
| **Random Forest** | **128 kWh** | **183 kWh** |
| Linear Regression | 310 kWh | 315 kWh |

The Random Forest achieves ~3–4% average error relative to mean daily load — suitable for grid planning purposes.

---

## 🖥️ Interactive Prediction Interface

Run `python energy_system.py` or call `run_interactive_interface()` in the notebook.

```
╔══════════════════════════════════════════════════════╗
║   🏙️  City Energy Consumption Prediction System      ║
║      Next-Day Demand Forecast Interface              ║
╚══════════════════════════════════════════════════════╝

  Available Zones:
    [1] Z1_Downtown
    [2] Z2_Residential
    ...

  Enter zone number: 1
  Enter tomorrow's avg temperature (°C): 32
  Enter tomorrow's humidity (%): 70
  Special event tomorrow? (0/1): 1

┌────────────────────────────────────────────┐
│  Zone   : Z1_Downtown                      │
│  Temp   :  32.0 °C                         │
│  Humidity:  70.0 %                         │
│  Event  :  YES 🎉                          │
├────────────────────────────────────────────┤
│  ⚡ Predicted Consumption:    7,342.5 kWh  │
└────────────────────────────────────────────┘
```

**Input validation** handles:
- Zone numbers outside valid range
- Non-numeric entries
- Temperature out of −10 to 50 °C range
- Humidity out of 10–100 % range
- Invalid event flag (not 0 or 1)

---

## 🛠️ Technical Stack

- **Python 3.10+**
- `numpy`, `pandas` – Data generation & manipulation
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – ML (Random Forest, Linear Regression, MAE evaluation)
- `pickle` – Model persistence

---

## 📄 License

MIT License — free to use, modify, and distribute.
