"""
City Energy Consumption Analysis & Prediction System
=====================================================
Generates synthetic data, analyzes patterns, visualizes trends,
trains an ML model, and provides an interactive prediction interface.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ── Styling ────────────────────────────────────────────────────────────────────
PALETTE = {
    'bg':        '#0F1117',
    'panel':     '#1A1D27',
    'accent1':   '#4F8EF7',
    'accent2':   '#F7934F',
    'accent3':   '#4FF7A8',
    'accent4':   '#F74F8E',
    'accent5':   '#C44FF7',
    'text':      '#E8EAF0',
    'subtext':   '#8B90A0',
    'grid':      '#262A38',
}
ZONE_COLORS = [PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent3'],
               PALETTE['accent4'], PALETTE['accent5']]
ZONES = ['Z1_Downtown', 'Z2_Industrial', 'Z3_Residential', 'Z4_Commercial', 'Z5_Airport']


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for 365 days × 5 zones.

    Zone consumption characteristics (kWh base):
      Downtown   – moderate residential/office mix
      Industrial – high, less weather-sensitive
      Residential – lower base, strong temperature & event sensitivity
      Commercial – moderate, strong event sensitivity
      Airport    – high and stable

    Returns a tidy DataFrame with columns:
        Date, ZoneID, AvgTemperature, Humidity, SpecialEvent, EnergyConsumption
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')

    # --- Zone baseline & sensitivity parameters ---
    zone_params = {
        'Z1_Downtown':   dict(base=18_000, temp_coef=120,  humid_coef=40,  event_boost=2_500),
        'Z2_Industrial': dict(base=45_000, temp_coef=60,   humid_coef=20,  event_boost=3_000),
        'Z3_Residential':dict(base=12_000, temp_coef=200,  humid_coef=80,  event_boost=3_500),
        'Z4_Commercial': dict(base=22_000, temp_coef=150,  humid_coef=50,  event_boost=5_000),
        'Z5_Airport':    dict(base=38_000, temp_coef=50,   humid_coef=15,  event_boost=4_000),
    }

    records = []
    for date in dates:
        doy = date.day_of_year
        month = date.month

        # Seasonal temperature: warm summers, cold winters (Mediterranean-ish city)
        base_temp = 15 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)
        temperature = base_temp + rng.normal(0, 3)

        # Humidity inversely correlated with temperature in summer
        base_humid = 65 - 15 * np.sin(2 * np.pi * (doy - 80) / 365)
        humidity = float(np.clip(base_humid + rng.normal(0, 8), 20, 100))

        # Special events: ~10% of days, more likely on weekends & summer
        is_weekend = date.weekday() >= 5
        event_prob = 0.10 + (0.06 if is_weekend else 0) + (0.04 if month in [6, 7, 8] else 0)
        special_event = int(rng.random() < event_prob)

        # Weekend / weekday effect on consumption
        weekday_factor = 0.85 if is_weekend else 1.0

        for zone, p in zone_params.items():
            # Core consumption driven by temperature (U-shape: heat & cold both raise usage)
            temp_deviation = abs(temperature - 18)          # comfort set-point = 18 °C
            consumption = (
                p['base'] * weekday_factor
                + p['temp_coef'] * temp_deviation
                + p['humid_coef'] * (humidity - 50)         # discomfort above 50 %
                + p['event_boost'] * special_event
                + rng.normal(0, p['base'] * 0.03)           # random noise ≈ 3 %
            )
            records.append({
                'Date':              date,
                'ZoneID':            zone,
                'AvgTemperature':    round(float(temperature), 2),
                'Humidity':          round(humidity, 2),
                'SpecialEvent':      special_event,
                'EnergyConsumption': max(0, round(float(consumption), 1)),
            })

    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRE-PROCESSING & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data, engineers time features, handles anomalies."""
    df = df.copy()

    # Drop exact duplicates
    df.drop_duplicates(inplace=True)

    # Fill any remaining NaNs with zone-level median
    num_cols = ['AvgTemperature', 'Humidity', 'EnergyConsumption']
    for col in num_cols:
        df[col] = df.groupby('ZoneID')[col].transform(
            lambda x: x.fillna(x.median()))

    # IQR-based outlier clipping per zone
    for col in num_cols:
        q1 = df.groupby('ZoneID')[col].transform(lambda x: x.quantile(0.25))
        q3 = df.groupby('ZoneID')[col].transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr)

    # Time-based features
    df['Month']      = df['Date'].dt.month
    df['DayOfWeek']  = df['Date'].dt.dayofweek        # 0=Mon … 6=Sun
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
    df['Quarter']    = df['Date'].dt.quarter
    df['DayOfYear']  = df['Date'].dt.day_of_year

    # Encode zone as integer
    le = LabelEncoder()
    df['ZoneEncoded'] = le.fit_transform(df['ZoneID'])
    df['ZoneLabel'] = le.classes_[df['ZoneEncoded']]   # round-trip check

    return df


def monthly_zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns average monthly consumption per zone."""
    return (df.groupby(['ZoneID', 'Month'])['EnergyConsumption']
              .mean()
              .reset_index()
              .rename(columns={'EnergyConsumption': 'AvgConsumption'}))


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between numeric features and EnergyConsumption."""
    features = ['AvgTemperature', 'Humidity', 'SpecialEvent',
                'IsWeekend', 'Month', 'EnergyConsumption']
    return df[features].corr()


def event_vs_nonevent(df: pd.DataFrame) -> pd.DataFrame:
    """Average consumption on event vs non-event days, per zone."""
    return (df.groupby(['ZoneID', 'SpecialEvent'])['EnergyConsumption']
              .mean()
              .reset_index())


# ══════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_dark_style(fig, axes_list):
    """Applies consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in axes_list:
        ax.set_facecolor(PALETTE['panel'])
        ax.tick_params(colors=PALETTE['text'], labelsize=9)
        ax.xaxis.label.set_color(PALETTE['text'])
        ax.yaxis.label.set_color(PALETTE['text'])
        ax.title.set_color(PALETTE['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE['grid'])
        ax.grid(color=PALETTE['grid'], linestyle='--', linewidth=0.5, alpha=0.7)


def plot_monthly_trends(df: pd.DataFrame, out_path: str):
    """Line chart: monthly average energy usage per zone."""
    summary = monthly_zone_summary(df)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(13, 6))
    _apply_dark_style(fig, [ax])

    for i, zone in enumerate(ZONES):
        zdata = summary[summary['ZoneID'] == zone].sort_values('Month')
        ax.plot(zdata['Month'], zdata['AvgConsumption'] / 1_000,
                color=ZONE_COLORS[i], linewidth=2.2, marker='o',
                markersize=6, markerfacecolor=PALETTE['bg'],
                markeredgewidth=2, label=zone, zorder=3)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, color=PALETTE['text'])
    ax.set_xlabel('Month', labelpad=10)
    ax.set_ylabel('Avg Energy Consumption (MWh)', labelpad=10)
    ax.set_title('Monthly Energy Consumption Trends by Zone', pad=16,
                 fontsize=14, fontweight='bold', color=PALETTE['text'])
    ax.legend(framealpha=0, labelcolor=PALETTE['text'], fontsize=9,
              loc='upper left')

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str):
    """Heatmap: feature correlations."""
    corr = correlation_matrix(df)
    labels = ['Temp (°C)', 'Humidity (%)', 'Special Event',
              'Is Weekend', 'Month', 'Energy (kWh)']

    fig, ax = plt.subplots(figsize=(9, 7))
    _apply_dark_style(fig, [ax])

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    cmap = sns.diverging_palette(220, 15, s=80, l=45, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor=PALETTE['bg'],
                ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 10, 'color': PALETTE['text']},
                vmin=-1, vmax=1)

    ax.set_title('Feature Correlation Matrix', pad=16,
                 fontsize=14, fontweight='bold', color=PALETTE['text'])
    ax.tick_params(axis='x', rotation=30, colors=PALETTE['text'])
    ax.tick_params(axis='y', rotation=0,  colors=PALETTE['text'])

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=PALETTE['text'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['text'])

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_event_vs_nonevent(df: pd.DataFrame, out_path: str):
    """Bar chart: avg consumption on event vs non-event days."""
    ev = event_vs_nonevent(df)
    zone_order = ZONES
    x = np.arange(len(zone_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_style(fig, [ax])

    for idx, (se, label, color) in enumerate([
        (0, 'No Event',      PALETTE['accent1']),
        (1, 'Special Event', PALETTE['accent2']),
    ]):
        vals = [ev[(ev['ZoneID'] == z) & (ev['SpecialEvent'] == se)]['EnergyConsumption'].values
                for z in zone_order]
        heights = [v[0] / 1_000 if len(v) else 0 for v in vals]
        bars = ax.bar(x + idx * width - width / 2, heights,
                      width=width, color=color, label=label,
                      edgecolor=PALETTE['bg'], linewidth=0.8, zorder=3)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom',
                    fontsize=8.5, color=PALETTE['text'])

    ax.set_xticks(x)
    ax.set_xticklabels([z.replace('_', '\n') for z in zone_order],
                       color=PALETTE['text'], fontsize=9)
    ax.set_ylabel('Avg Energy Consumption (MWh)', labelpad=10)
    ax.set_title('Energy Consumption: Event vs Non-Event Days by Zone',
                 pad=16, fontsize=14, fontweight='bold', color=PALETTE['text'])
    ax.legend(framealpha=0, labelcolor=PALETTE['text'])

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_model_performance(y_test, y_pred, zone_labels, out_path: str):
    """Scatter plot of actual vs predicted values, coloured by zone."""
    zone_to_color = {z: ZONE_COLORS[i] for i, z in enumerate(ZONES)}
    colors = [zone_to_color.get(z, PALETTE['accent1']) for z in zone_labels]

    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_dark_style(fig, [ax])

    ax.scatter(np.array(y_test) / 1_000, np.array(y_pred) / 1_000,
               c=colors, alpha=0.55, s=18, edgecolors='none', zorder=3)

    lims = [min(np.min(y_test), np.min(y_pred)) / 1_000 * 0.97,
            max(np.max(y_test), np.max(y_pred)) / 1_000 * 1.03]
    ax.plot(lims, lims, '--', color=PALETTE['accent3'], linewidth=1.5,
            label='Perfect prediction', zorder=4)

    # Legend patches per zone
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ZONE_COLORS[i], label=ZONES[i]) for i in range(len(ZONES))]
    handles.append(plt.Line2D([0], [0], linestyle='--',
                              color=PALETTE['accent3'], label='Perfect prediction'))
    ax.legend(handles=handles, framealpha=0, labelcolor=PALETTE['text'], fontsize=9)

    ax.set_xlabel('Actual Consumption (MWh)', labelpad=10)
    ax.set_ylabel('Predicted Consumption (MWh)', labelpad=10)
    ax.set_title('Random Forest: Actual vs Predicted', pad=16,
                 fontsize=14, fontweight='bold', color=PALETTE['text'])

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ML MODEL
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ['ZoneEncoded', 'AvgTemperature', 'Humidity',
            'SpecialEvent', 'Month', 'DayOfWeek', 'IsWeekend']
TARGET   = 'EnergyConsumption'


def train_models(df: pd.DataFrame):
    """
    Trains both LinearRegression and RandomForest.
    Returns (rf_model, lr_model, X_test, y_test, zone_test, results_dict).
    """
    X = df[FEATURES]
    y = df[TARGET]
    zones = df['ZoneID']

    X_train, X_test, y_train, y_test, _, z_test = train_test_split(
        X, y, zones, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae  = mean_absolute_error(y_test, lr_pred)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae  = mean_absolute_error(y_test, rf_pred)

    results = {
        'rf_mae': rf_mae,
        'lr_mae': lr_mae,
        'rf_pred': rf_pred,
        'lr_pred': lr_pred,
    }
    return rf, lr, X_test, y_test.values, z_test.values, results


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERACTIVE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def interactive_predictor(rf_model, le_zones: dict):
    """
    Command-line interface: user inputs zone + weather, system predicts kWh.
    `le_zones` maps zone name → encoded integer.
    """
    ZONE_NAMES = list(le_zones.keys())

    print("\n" + "═" * 58)
    print("  🏙  City Energy Consumption Predictor")
    print("═" * 58)

    while True:
        print("\nAvailable Zones:")
        for i, z in enumerate(ZONE_NAMES, 1):
            print(f"  {i}. {z}")
        zone_input = input("\nEnter Zone ID (name or number) [or 'q' to quit]: ").strip()

        if zone_input.lower() in ('q', 'quit', 'exit'):
            print("👋  Goodbye!")
            break

        # Resolve zone
        zone_name = None
        if zone_input.isdigit():
            idx = int(zone_input) - 1
            if 0 <= idx < len(ZONE_NAMES):
                zone_name = ZONE_NAMES[idx]
        elif zone_input in ZONE_NAMES:
            zone_name = zone_input
        else:
            # Partial match
            matches = [z for z in ZONE_NAMES if zone_input.lower() in z.lower()]
            if len(matches) == 1:
                zone_name = matches[0]

        if zone_name is None:
            print("⚠  Invalid zone. Please try again.")
            continue

        # Temperature
        try:
            temp = float(input("Enter tomorrow's average temperature (°C, e.g. 22.5): "))
            if not -30 <= temp <= 55:
                raise ValueError("Temperature out of realistic range.")
        except ValueError as e:
            print(f"⚠  {e}")
            continue

        # Humidity
        try:
            humidity = float(input("Enter tomorrow's humidity (%, 0-100): "))
            if not 0 <= humidity <= 100:
                raise ValueError("Humidity must be between 0 and 100.")
        except ValueError as e:
            print(f"⚠  {e}")
            continue

        # Event
        try:
            event_raw = input("Special event tomorrow? (0 = No, 1 = Yes): ").strip()
            if event_raw not in ('0', '1'):
                raise ValueError("Please enter 0 or 1.")
            event = int(event_raw)
        except ValueError as e:
            print(f"⚠  {e}")
            continue

        # Month / day (use tomorrow's calendar info)
        import datetime
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        month      = tomorrow.month
        day_of_week = tomorrow.weekday()
        is_weekend  = int(day_of_week >= 5)

        zone_enc = le_zones[zone_name]
        X_pred = pd.DataFrame([{
            'ZoneEncoded':    zone_enc,
            'AvgTemperature': temp,
            'Humidity':       humidity,
            'SpecialEvent':   event,
            'Month':          month,
            'DayOfWeek':      day_of_week,
            'IsWeekend':      is_weekend,
        }])

        prediction = rf_model.predict(X_pred)[0]

        print("\n" + "─" * 58)
        print(f"  Zone            : {zone_name}")
        print(f"  Date            : {tomorrow.strftime('%A, %d %b %Y')}")
        print(f"  Temperature     : {temp} °C")
        print(f"  Humidity        : {humidity} %")
        print(f"  Special Event   : {'Yes' if event else 'No'}")
        print(f"\n  ⚡ Predicted Consumption: {prediction:,.0f} kWh  "
              f"({prediction/1000:.2f} MWh)")
        print("─" * 58)

        again = input("\nPredict for another zone? (y/n): ").strip().lower()
        if again != 'y':
            print("👋  Goodbye!")
            break


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(output_dir: str = '.', interactive: bool = False):
    """End-to-end pipeline: generate → preprocess → analyse → visualise → model."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n🔄  Step 1 — Generating synthetic dataset …")
    raw_df = generate_dataset()
    raw_df.to_csv(os.path.join(output_dir, 'energy_data_raw.csv'), index=False)
    print(f"    Shape: {raw_df.shape}  |  Date range: {raw_df['Date'].min().date()} "
          f"→ {raw_df['Date'].max().date()}")

    print("\n🔄  Step 2 — Preprocessing …")
    df = preprocess(raw_df)
    df.to_csv(os.path.join(output_dir, 'energy_data_clean.csv'), index=False)

    # Quick stats
    print("\n📊  Summary Statistics:")
    print(df.groupby('ZoneID')['EnergyConsumption'].describe()[
        ['mean','std','min','max']].round(0).to_string())

    print("\n📈  Correlation with EnergyConsumption:")
    corr = correlation_matrix(df)['EnergyConsumption'].drop('EnergyConsumption')
    for feat, val in corr.sort_values(key=abs, ascending=False).items():
        print(f"    {feat:<20} {val:+.3f}")

    print("\n🎨  Step 3 — Generating visualisations …")
    plot_monthly_trends(df,    os.path.join(output_dir, 'plot1_monthly_trends.png'))
    plot_correlation_heatmap(df, os.path.join(output_dir, 'plot2_correlation_heatmap.png'))
    plot_event_vs_nonevent(df, os.path.join(output_dir, 'plot3_event_comparison.png'))

    print("\n🤖  Step 4 — Training ML models …")
    rf, lr, X_test, y_test, z_test, results = train_models(df)
    print(f"    Linear Regression  MAE: {results['lr_mae']:,.0f} kWh")
    print(f"    Random Forest      MAE: {results['rf_mae']:,.0f} kWh  ✓ (best model)")

    plot_model_performance(y_test, results['rf_pred'], z_test,
                           os.path.join(output_dir, 'plot4_model_performance.png'))

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n    Feature Importances (Random Forest):")
    for feat, imp in fi.items():
        bar = '█' * int(imp * 60)
        print(f"    {feat:<18} {bar}  {imp:.3f}")

    # Save model
    model_path = os.path.join(output_dir, 'rf_model.joblib')
    joblib.dump(rf, model_path)
    print(f"\n    Model saved → {model_path}")

    # Zone encoding map
    le_zones = {zone: i for i, zone in enumerate(sorted(ZONES))}

    if interactive:
        print("\n🖥  Step 5 — Interactive Predictor")
        interactive_predictor(rf, le_zones)

    return df, rf, results


if __name__ == '__main__':
    run_pipeline(output_dir='outputs', interactive=True)
