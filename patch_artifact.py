"""
patch_artifact.py
-----------------
Replicates feature engineering + scaling from planner.ipynb,
then re-saves productivity_model.pkl with the fitted StandardScaler.

Run once from the project directory:
  python patch_artifact.py
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 1. Load raw CSVs ──────────────────────────────────────────────────────────
print("Loading data...")
period  = pd.read_csv("Period_Log.csv")
profile = pd.read_csv("User_Profile.csv")

# ── 2. Merge on user_id ───────────────────────────────────────────────────────
data = pd.merge(period, profile, on="user_id", how="inner")
print(f"Rows after merge: {len(data)}")

# ── 3. Drop nulls ─────────────────────────────────────────────────────────────
data.dropna(inplace=True)
print(f"Rows after dropna: {len(data)}")

# ── 4. Encode categorical features ───────────────────────────────────────────
data['pms_symptoms'] = data['pms_symptoms'].map({'Yes': 1, 'No': 0})
data['flow_level']   = data['flow_level'].map({'Heavy': 3, 'Moderate': 2, 'Light': 1})

# ── 5. One-hot encode cycle_phase and ovulation_result ───────────────────────
data = pd.get_dummies(data, columns=['cycle_phase', 'ovulation_result'])

# ── 6. Pick the right 'sleep_hours' column ───────────────────────────────────
#   Period_Log has 'sleep_hours_cycle'; User_Profile has 'sleep_hours'
#   After merge both exist with a suffix if names clash:
#   → if 'sleep_hours' exists (from profile), use it; else fall back to cycle
if 'sleep_hours' not in data.columns and 'sleep_hours_cycle' in data.columns:
    data.rename(columns={'sleep_hours_cycle': 'sleep_hours'}, inplace=True)
    print("Renamed 'sleep_hours_cycle' → 'sleep_hours'")

print("Columns available:", [c for c in data.columns if 'sleep' in c.lower()])

# ── 7. Select the exact 18 training features ─────────────────────────────────
FEATURES = [
    'cycle_length_days', 'flow_level', 'pain_level', 'pms_symptoms',
    'mood_score', 'stress_score_cycle', 'energy_level', 'concentration_score',
    'work_hours_lost', 'estrogen_pgml', 'progesterone_ngml', 'sleep_hours',
    'pcos_diagnosed', 'cycle_phase_Follicular', 'cycle_phase_Luteal',
    'cycle_phase_Menstrual', 'ovulation_result_Negative', 'ovulation_result_Positive'
]

available = [c for c in FEATURES if c in data.columns]
missing   = [c for c in FEATURES if c not in data.columns]

if missing:
    print(f"⚠️  Missing features: {missing}")
    print("All columns:", list(data.columns))
else:
    print("✅  All 18 features found.")

x = data[available]

# ── 8. Fit StandardScaler ─────────────────────────────────────────────────────
print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(x)

# ── 9. Load artifact, inject scaler, re-save ─────────────────────────────────
print("Updating artifact...")
with open("productivity_model.pkl", "rb") as f:
    artifact = pickle.load(f)

artifact["scaler"]   = scaler
artifact["features"] = available

with open("productivity_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("✅  productivity_model.pkl updated successfully!")
print(f"   Features saved: {len(available)}")
print(f"   Scaler means (first 5): {scaler.mean_[:5].round(3)}")
print(f"   Scaler stds  (first 5): {scaler.scale_[:5].round(3)}")
