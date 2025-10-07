"""
═══════════════════════════════════════════════════════════════════════════════
🏆 IMPROVED ONE-FILE CHAMPION EXPERIMENT 🏆
═══════════════════════════════════════════════════════════════════════════════

Goal: Implement the suggested modeling upgrades BEFORE final blending.
This script is an experimental enhancement of `generate_champion_complete.py`.

Key Improvements vs Original Champion Script:
1. Delayed Rounding: Base model predictions kept as floats; only final blend
   is clipped/rounded → preserves variance for a better blend.
2. Expanded Alpha Searches:
   • Model 1: Broader alpha grid per scaler.
   • Model 2 Submodels: Tuned separately (shared fixed alpha for now but easily extendable).
   • Model 3: Fine grid near previously strong region (0.2–0.5).
3. Multi-Seed Expansion: Model 3 uses 5 seeds (42,123,456,789,2024).
4. Proper OOF Framework:
   • Generates out-of-fold predictions for each of the 3 final models.
   • Grid search for blend weights using OOF MAE (simple constrained search).
5. Model 2 Internal Weighting:
   • Produces OOF predictions for its two submodels once, then searches weights.
6. Single Final Rounding.

Optional Future Additions (NOT enabled yet to isolate improvements):
   • Win% target transformation (predict W/G, then * G).
   • Interaction & ratio feature expansions.
   • Continuous optimization (coordinate descent) for blend weights.

Output:
  • CSV: submission_champion_improved.csv
  • Printed diagnostics for each model + blend.

Status: EXPERIMENTAL (intended to beat or match original 2.97530 MAE expectation)
Date: October 7, 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_TRAIN = 'data/train.csv'
DATA_TEST = 'data/test.csv'
OUTPUT_FILE = 'submission_champion_improved.csv'
N_FOLDS = 10
RANDOM_STATE = 42

# Blend weight search ranges (w1=Model1, w2=Model2, w3=Model3)
BLEND_GRID_W1 = np.arange(0.25, 0.56, 0.02)
BLEND_GRID_W2 = np.arange(0.25, 0.56, 0.02)
MIN_W3, MAX_W3 = 0.10, 0.40

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('📊 LOADING DATA (Improved Experiment)')
print('='*80)
train_df = pd.read_csv(DATA_TRAIN)
test_df = pd.read_csv(DATA_TEST)
y = train_df['W'].astype(float)
print(f'Train shape: {train_df.shape} | Test shape: {test_df.shape}')
print()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING HELPERS (copied + modular tweaks)
# ══════════════════════════════════════════════════════════════════════════════

def create_stable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Base feature generation excluding explicit temporal flags.
    Same logic as original champion; can be extended safely later."""
    df = df.copy()
    if {'R','RA','G'}.issubset(df.columns):
        for exp in [1.83, 1.85, 1.9, 2.0]:
            exp_str = str(int(exp*100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / (df['G'] + 1)
        df['run_ratio'] = df['R'] / (df['RA'] + 1)

    if 'G' in df.columns:
        for col in ['R','RA','H','HR','BB','SO']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / (df['G'] + 1)

    if {'H','AB'}.issubset(df.columns):
        df['BA'] = df['H'] / (df['AB'] + 1)
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        if {'2B','3B','HR'}.issubset(df.columns):
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            if 'OBP' in df.columns:
                df['OPS'] = df['OBP'] + df['SLG']

    if {'ERA','IPouts'}.issubset(df.columns):
        if {'HA','BBA'}.issubset(df.columns):
            df['WHIP'] = (df['HA'] + df['BBA']) / ((df['IPouts']/3) + 1)
        if 'SOA' in df.columns:
            df['K_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 1)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum():
            df[col] = df[col].fillna(df[col].median())
    return df

def create_feature_set_1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'R','RA','G'}.issubset(df.columns):
        for exp in [1.83, 2.0]:
            exp_str = str(int(exp*100))
            df[f'pyth_wins_{exp_str}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp + 1)) * df['G']
        df['run_diff_per_game'] = (df['R'] - df['RA']) / (df['G'] + 1)
    return df

def create_feature_set_2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'G' in df.columns:
        for col in ['R','RA']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / (df['G'] + 1)
    if {'H','AB','BB','2B','3B','HR'}.issubset(df.columns):
        singles = df['H'] - df['2B'] - df['3B'] - df['HR']
        df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
        df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
    return df

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf,-np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum():
            df[col] = df[col].fillna(df[col].median())
    return df

EXCLUDE_BASE = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910','decade_1920','decade_1930','decade_1940','decade_1950',
    'decade_1960','decade_1970','decade_1980','decade_1990','decade_2000','decade_2010',
    'era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','mlb_rpg'
}

EXCLUDE_MULTI = EXCLUDE_BASE | {
    'H','AB','R','RA','HR','BB','2B','3B','SO','HA','HRA','BBA','SOA','E','DP','SB','FP'
}

# ══════════════════════════════════════════════════════════════════════════════
# KFold
# ══════════════════════════════════════════════════════════════════════════════
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: NO-TEMPORAL (Improved)
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('MODEL 1: NO-TEMPORAL (Improved alpha + scaler grid, no early rounding)')
print('='*80)
train_notemporal = create_stable_features(train_df)
test_notemporal = create_stable_features(test_df)

feat1 = sorted((set(train_notemporal.columns) & set(test_notemporal.columns)) - EXCLUDE_BASE)
X1_train = train_notemporal[feat1].fillna(0)
X1_test = test_notemporal[feat1].fillna(0)

scalers = [StandardScaler(), RobustScaler()]
alpha_grid_m1 = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

best_m1 = {
    'mae': np.inf,
    'scaler': None,
    'alpha': None
}

for sc in scalers:
    X_scaled = sc.fit_transform(X1_train)
    for a in alpha_grid_m1:
        ridge = Ridge(alpha=a)
        scores = cross_val_score(ridge, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -scores.mean()
        if mae < best_m1['mae']:
            best_m1.update({'mae': mae, 'scaler': sc, 'alpha': a})

print(f"✓ Best Model1 alpha={best_m1['alpha']} scaler={best_m1['scaler'].__class__.__name__} CV_MAE={best_m1['mae']:.4f}")

X1_train_scaled = best_m1['scaler'].fit_transform(X1_train)
X1_test_scaled = best_m1['scaler'].transform(X1_test)
model1_final = Ridge(alpha=best_m1['alpha']).fit(X1_train_scaled, y)
pred_m1_test = model1_final.predict(X1_test_scaled)  # float predictions retained

# OOF for Model 1
oof_m1 = np.zeros(len(y))
for tr_idx, val_idx in kfold.split(X1_train_scaled):
    m = Ridge(alpha=best_m1['alpha'])
    m.fit(X1_train_scaled[tr_idx], y.iloc[tr_idx])
    oof_m1[val_idx] = m.predict(X1_train_scaled[val_idx])

print(f"Model 1 OOF MAE: {mean_absolute_error(y, oof_m1):.4f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: MULTI-ENSEMBLE (Improved OOF weight search)
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('MODEL 2: MULTI-ENSEMBLE (OOF-based weight tuning)')
print('='*80)
train_s1 = clean_numeric(create_feature_set_1(train_df))
train_s2 = clean_numeric(create_feature_set_2(train_df))

test_s1 = clean_numeric(create_feature_set_1(test_df))
test_s2 = clean_numeric(create_feature_set_2(test_df))

features_s1 = sorted((set(train_s1.columns) & set(test_s1.columns)) - EXCLUDE_MULTI)
features_s2 = sorted((set(train_s2.columns) & set(test_s2.columns)) - EXCLUDE_MULTI)

X2a_train = train_s1[features_s1].fillna(0)
X2b_train = train_s2[features_s2].fillna(0)
X2a_test = test_s1[features_s1].fillna(0)
X2b_test = test_s2[features_s2].fillna(0)

scaler_2a = StandardScaler().fit(X2a_train)
scaler_2b = StandardScaler().fit(X2b_train)
X2a_train_scaled = scaler_2a.transform(X2a_train)
X2b_train_scaled = scaler_2b.transform(X2b_train)
X2a_test_scaled = scaler_2a.transform(X2a_test)
X2b_test_scaled = scaler_2b.transform(X2b_test)

alpha_sub = 3.0  # Keep single alpha; can grid search individually if desired

# Build OOF predictions for each submodel once
print('Generating OOF predictions for submodels...')
oof_sub1 = np.zeros(len(y))
oof_sub2 = np.zeros(len(y))
for tr_idx, val_idx in kfold.split(X2a_train_scaled):
    m_a = Ridge(alpha=alpha_sub)
    m_a.fit(X2a_train_scaled[tr_idx], y.iloc[tr_idx])
    oof_sub1[val_idx] = m_a.predict(X2a_train_scaled[val_idx])

    m_b = Ridge(alpha=alpha_sub)
    m_b.fit(X2b_train_scaled[tr_idx], y.iloc[tr_idx])
    oof_sub2[val_idx] = m_b.predict(X2b_train_scaled[val_idx])

# Weight search for internal ensemble
def search_internal_weights(oof1, oof2, target):
    best = {'w1': None, 'mae': np.inf}
    for w1 in np.arange(0.30, 0.71, 0.05):
        blend = w1 * oof1 + (1 - w1) * oof2
        mae = mean_absolute_error(target, blend)
        if mae < best['mae']:
            best.update({'w1': w1, 'mae': mae})
    return best

best_internal = search_internal_weights(oof_sub1, oof_sub2, y)
print(f"✓ Internal weights: w_sub1={best_internal['w1']:.2f} w_sub2={1-best_internal['w1']:.2f} OOF_MAE={best_internal['mae']:.4f}")

# Fit final submodels on full data
model2_sub1 = Ridge(alpha=alpha_sub).fit(X2a_train_scaled, y)
model2_sub2 = Ridge(alpha=alpha_sub).fit(X2b_train_scaled, y)

pred_sub1_test = model2_sub1.predict(X2a_test_scaled)
pred_sub2_test = model2_sub2.predict(X2b_test_scaled)

pred_m2_test = best_internal['w1'] * pred_sub1_test + (1 - best_internal['w1']) * pred_sub2_test

# OOF for final Model 2 combination
oof_m2 = best_internal['w1'] * oof_sub1 + (1 - best_internal['w1']) * oof_sub2
print(f"Model 2 OOF MAE: {mean_absolute_error(y, oof_m2):.4f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: FINE-TUNED (Expanded alpha + 5-seed ensemble)
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('MODEL 3: FINE-TUNED (alpha grid + 5 seeds, OOF)')
print('='*80)
train_ft = create_stable_features(train_df)
test_ft = create_stable_features(test_df)

features_ft = sorted((set(train_ft.columns) & set(test_ft.columns)) - EXCLUDE_BASE)
X3_train = train_ft[features_ft].fillna(0)
X3_test = test_ft[features_ft].fillna(0)

scaler3 = RobustScaler().fit(X3_train)
X3_train_scaled = scaler3.transform(X3_train)
X3_test_scaled = scaler3.transform(X3_test)

alpha_grid_ft = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

best_ft = {'alpha': None, 'mae': np.inf}
for a in alpha_grid_ft:
    ridge = Ridge(alpha=a, random_state=RANDOM_STATE)
    scores = cross_val_score(ridge, X3_train_scaled, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_ft['mae']:
        best_ft.update({'alpha': a, 'mae': mae})
print(f"✓ Best fine-tuned alpha={best_ft['alpha']} CV_MAE={best_ft['mae']:.4f}")

# Multi-seed ensemble predictions (test)
seeds_ft = [42, 123, 456, 789, 2024]
pred_seed_list = []
for sd in seeds_ft:
    m = Ridge(alpha=best_ft['alpha'], random_state=sd)
    m.fit(X3_train_scaled, y)
    pred_seed_list.append(m.predict(X3_test_scaled))

pred_m3_test = np.mean(pred_seed_list, axis=0)

# Build OOF predictions for Model 3 (use 2 seeds for speed inside folds)
oof_m3 = np.zeros(len(y))
for tr_idx, val_idx in kfold.split(X3_train_scaled):
    fold_preds = []
    for sd in [42, 123]:
        m = Ridge(alpha=best_ft['alpha'], random_state=sd)
        m.fit(X3_train_scaled[tr_idx], y.iloc[tr_idx])
        fold_preds.append(m.predict(X3_train_scaled[val_idx]))
    oof_m3[val_idx] = np.mean(fold_preds, axis=0)
print(f"Model 3 OOF MAE: {mean_absolute_error(y, oof_m3):.4f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# BLEND WEIGHT SEARCH (Using OOF predictions of the THREE models)
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('🏗  GLOBAL BLEND WEIGHT GRID SEARCH (OOF MAE)')
print('='*80)

# Candidate weight grid
best_blend = {
    'w1': 0.37,
    'w2': 0.44,
    'w3': 0.19,
    'mae': mean_absolute_error(y, 0.37*oof_m1 + 0.44*oof_m2 + 0.19*oof_m3)
}

total_evals = 0
for w1 in BLEND_GRID_W1:
    for w2 in BLEND_GRID_W2:
        w3 = 1 - w1 - w2
        if w3 < 0:  # invalid
            continue
        if not (MIN_W3 <= w3 <= MAX_W3):
            continue
        blend = w1*oof_m1 + w2*oof_m2 + w3*oof_m3
        mae = mean_absolute_error(y, blend)
        if mae < best_blend['mae']:
            best_blend.update({'w1': w1, 'w2': w2, 'w3': w3, 'mae': mae})
        total_evals += 1

print(f"Evaluations performed: {total_evals}")
print(f"✓ Best OOF blend: w1={best_blend['w1']:.2f} w2={best_blend['w2']:.2f} w3={best_blend['w3']:.2f} MAE={best_blend['mae']:.4f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL PREDICTION & SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('🏁 CREATING FINAL PREDICTIONS (Single rounding)')
print('='*80)

pred_final_float = (
    best_blend['w1'] * pred_m1_test +
    best_blend['w2'] * pred_m2_test +
    best_blend['w3'] * pred_m3_test
)

pred_final = np.clip(pred_final_float, 0, 162).round().astype(int)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'W': pred_final
})
submission.to_csv(OUTPUT_FILE, index=False)

print('Final prediction stats:')
print(f"  Min:  {pred_final.min()}")
print(f"  Max:  {pred_final.max()}")
print(f"  Mean: {pred_final.mean():.2f}")
print(f"  Std:  {pred_final.std():.2f}")
print()
print(f"Saved submission: {OUTPUT_FILE} (rows={len(submission)})")
print()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print('='*80)
print('✅ IMPROVED EXPERIMENT COMPLETE')
print('='*80)
print('Model OOF MAEs:')
print(f"  • Model 1 (No-Temporal): {mean_absolute_error(y, oof_m1):.4f}")
print(f"  • Model 2 (Multi-Ensemble): {mean_absolute_error(y, oof_m2):.4f}")
print(f"  • Model 3 (Fine-Tuned): {mean_absolute_error(y, oof_m3):.4f}")
print()
print('Blend:')
print(f"  • Weights: {best_blend['w1']:.2f}/{best_blend['w2']:.2f}/{best_blend['w3']:.2f}")
print(f"  • OOF Blend MAE: {best_blend['mae']:.4f}")
print()
print('Notes:')
print('  - Further gains: add win% target transform, ratio/interactions, refine alpha per submodel.')
print('  - You can tighten blend grid or implement coordinate descent for precision.')
print('  - Keep this file separate for reproducible comparison vs original champion.')
print()
print('🚀 Ready to evaluate on Kaggle (compare to original 2.97530).')
print('='*80)
