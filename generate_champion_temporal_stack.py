"""
═══════════════════════════════════════════════════════════════════════════════
TEMPORAL STACKING MODEL (Chronological CV + Meta Ridge)
═══════════════════════════════════════════════════════════════════════════════
Motivation:
  Recent leaderboard score worsened to ~3.12 despite low random-KFold OOF (~2.77).
  High adversarial AUC (~1.0) confirms extreme distribution shift (temporal).

Approach:
  1. Use chronological expanding-window folds (N_CHRONO_SPLITS) to generate
     out-of-fold (OOF) predictions for multiple base models trained ONLY on
     past data to predict future eras.
     Base models:
       • R1: Stable feature Ridge (moderate alpha grid)
       • R2: Full feature Ridge (possibly more variance) with stronger alphas
       • EN: ElasticNet (sparse, may reduce overfit to non-stationary noise)
       • TR: Shallow HistGradientBoostingRegressor (non-linear, limited depth)
  2. Collect OOF predictions (shape: n_samples x n_base_models).
  3. Train meta ridge (second level) on OOF matrix vs y using chronological
     folds inside for alpha selection (meta emphasizes models that generalize
     across eras).
  4. Refit each base model on full training data (with chosen hyperparameters)
     respecting feature scaling; produce test predictions.
  5. Blend test predictions with meta model (predict over stacked test matrix).
  6. Optional residual decade bias correction (skipped if test lacks yearID).

Design choices:
  • Chronological MAE used for model + alpha selection (not random folds).
  • Stronger alpha grid skewed toward higher values for stability.
  • ElasticNet narrow alpha + l1_ratio search to control complexity.
  • Tree model shallow (depth <=5) and early capped iterations.
  • Meta model uses ridge with its own small alpha grid.

Outputs:
  submission_champion_temporal_stack.csv
  temporal_stack_diagnostics.json (alphas, OOF MAEs, meta weights approximation)

Potential Next Enhancements:
  - Residual modeling on meta residuals (chronological) with tiny GB.
  - Era-weighted loss (weight recent eras more if closer to test era distribution).
  - Feature drift pruning layer before base models (integrate drift importances).

Run:
  python generate_champion_temporal_stack.py

Date: 2025-10-07
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json, warnings
from typing import List, Dict, Tuple
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST  = 'data/test.csv'
SUBMISSION_FILE = 'submission_champion_temporal_stack.csv'
DIAGNOSTICS_FILE = 'temporal_stack_diagnostics.json'

RANDOM_STATE = 42
N_CHRONO_SPLITS = 8
RIDGE_ALPHA_GRID_STABLE = [0.3,0.4,0.5,0.75,1.0,1.5,2.0,3.0]
RIDGE_ALPHA_GRID_FULL   = [0.5,0.75,1.0,1.5,2.5,4.0,6.0]
EN_ALPHA_GRID = [0.0005,0.001,0.003,0.005,0.01]
EN_L1_RATIOS  = [0.2,0.4,0.6,0.8]
TREE_PARAMS = dict(max_depth=5, learning_rate=0.07, max_iter=350,
                   min_samples_leaf=35, l2_regularization=0.8,
                   random_state=RANDOM_STATE)
META_ALPHA_GRID = [0.001,0.005,0.01,0.02,0.05,0.1]

DECADE_BIAS_CORRECTION = True
DECADE_CORR_SHRINK = 0.5
DECADE_CORR_MAX_ABS = 1.0

np.random.seed(RANDOM_STATE)

# =============================================================================
# DATA LOADING
# =============================================================================
train_df = pd.read_csv(DATA_TRAIN)
test_df  = pd.read_csv(DATA_TEST)
y = train_df['W'].astype(float)
if 'yearID' not in train_df.columns:
    raise ValueError('yearID column required for temporal stacking script.')
train_years = train_df['yearID'].values

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'R','RA','G'}.issubset(df.columns):
        for exp in [1.83,1.9,2.0]:
            exp_str = str(int(exp*100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / (df['G'] + 1)
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    if 'G' in df.columns:
        for col in ['R','RA','H','HR','BB','SO']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col]/(df['G']+1)
    if {'H','AB'}.issubset(df.columns):
        df['BA'] = df['H']/(df['AB']+1)
        if 'BB' in df.columns:
            df['OBP']=(df['H']+df['BB'])/(df['AB']+df['BB']+1)
        if {'2B','3B','HR'}.issubset(df.columns):
            singles = df['H']-df['2B']-df['3B']-df['HR']
            df['SLG']=(singles+2*df['2B']+3*df['3B']+4*df['HR'])/(df['AB']+1)
            if 'OBP' in df.columns:
                df['OPS']=df['OBP']+df['SLG']
    if {'ERA','IPouts'}.issubset(df.columns):
        if {'HA','BBA'}.issubset(df.columns):
            df['WHIP']=(df['HA']+df['BBA'])/((df['IPouts']/3)+1)
        if 'SOA' in df.columns:
            df['K_per_9']=(df['SOA']*27)/(df['IPouts']+1)
    df = df.replace([np.inf,-np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum():
            df[col] = df[col].fillna(df[col].median())
    return df

EXCLUDE = {
    'W','ID','teamID','yearID','year_label','decade_label','win_bins',
    'decade_1910','decade_1920','decade_1930','decade_1940','decade_1950',
    'decade_1960','decade_1970','decade_1980','decade_1990','decade_2000','decade_2010',
    'era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','mlb_rpg'
}

train_feat_full = build_features(train_df)
test_feat_full  = build_features(test_df)
all_features = sorted((set(train_feat_full.columns) & set(test_feat_full.columns)) - EXCLUDE)

X_full = train_feat_full[all_features].values
X_test_full = test_feat_full[all_features].values

# Stable feature subset = excluding raw heavy variance raw counts (heuristic)
stable_mask = [not f.startswith('R_') and not f.startswith('RA_') for f in all_features]
# If that heuristic yields too few, fallback to all
if sum(stable_mask) < 10:
    stable_mask = [True]*len(all_features)
stable_features = [f for f, keep in zip(all_features, stable_mask) if keep]
X_stable = train_feat_full[stable_features].values
X_test_stable = test_feat_full[stable_features].values

scaler_stable = RobustScaler().fit(X_stable)
X_stable_s = scaler_stable.transform(X_stable)
X_test_stable_s = scaler_stable.transform(X_test_stable)

scaler_full = RobustScaler().fit(X_full)
X_full_s = scaler_full.transform(X_full)
X_test_full_s = scaler_full.transform(X_test_full)

# ElasticNet needs StandardScaler typically
scaler_en = StandardScaler().fit(X_full)
X_en_s = scaler_en.transform(X_full)
X_test_en_s = scaler_en.transform(X_test_full)

# Chronological fold construction
sorted_year_idx = np.argsort(train_years)
years_sorted = train_years[sorted_year_idx]
unique_years = np.unique(years_sorted)
year_splits = np.array_split(unique_years, N_CHRONO_SPLITS)

# Helper to iterate folds
fold_definitions = []
for i in range(1, len(year_splits)):
    val_years = year_splits[i]
    train_years_concat = np.concatenate(year_splits[:i])
    fold_definitions.append((train_years_concat, val_years))

# =============================================================================
# FUNCTION: Chronological alpha CV for Ridge
# =============================================================================

def chrono_select_alpha(Xm: np.ndarray, alpha_grid: List[float]) -> Tuple[float, float]:
    Xm_sorted = Xm[sorted_year_idx]
    y_sorted = y.iloc[sorted_year_idx].reset_index(drop=True)
    best_alpha = None
    best_mae = np.inf
    for a in alpha_grid:
        fold_maes = []
        for train_years_concat, val_years in fold_definitions:
            tr_mask = np.isin(years_sorted, train_years_concat)
            va_mask = np.isin(years_sorted, val_years)
            if tr_mask.sum() == 0 or va_mask.sum() == 0:
                continue
            m = Ridge(alpha=a)
            m.fit(Xm_sorted[tr_mask], y_sorted[tr_mask])
            pred = m.predict(Xm_sorted[va_mask])
            fold_maes.append(mean_absolute_error(y_sorted[va_mask], pred))
        if fold_maes:
            mae = np.mean(fold_maes)
            if mae < best_mae:
                best_mae = mae
                best_alpha = a
    return best_alpha, best_mae

# =============================================================================
# BASE MODELS OOF COLLECTION
# =============================================================================
print('Selecting alphas (chronological)...')
alpha_stable, mae_stable = chrono_select_alpha(X_stable_s, RIDGE_ALPHA_GRID_STABLE)
alpha_full,   mae_full   = chrono_select_alpha(X_full_s,   RIDGE_ALPHA_GRID_FULL)
print(f"Stable ridge alpha={alpha_stable} chrono_MAE={mae_stable:.4f}")
print(f"Full   ridge alpha={alpha_full} chrono_MAE={mae_full:.4f}")

# ElasticNet alpha/l1 search (chronological simplified: evaluate small grid)
print('ElasticNet hyperparameter search (chronological)...')
best_en = {'alpha':None,'l1':None,'mae':np.inf}
for a in EN_ALPHA_GRID:
    for l1 in EN_L1_RATIOS:
        fold_maes = []
        for train_years_concat, val_years in fold_definitions:
            tr_mask = np.isin(train_years, train_years_concat)
            va_mask = np.isin(train_years, val_years)
            if tr_mask.sum()==0 or va_mask.sum()==0:
                continue
            en = ElasticNet(alpha=a, l1_ratio=l1, random_state=RANDOM_STATE, max_iter=5000)
            en.fit(X_en_s[tr_mask], y.iloc[tr_mask])
            pred = en.predict(X_en_s[va_mask])
            fold_maes.append(mean_absolute_error(y.iloc[va_mask], pred))
        if fold_maes:
            m_mae = np.mean(fold_maes)
            if m_mae < best_en['mae']:
                best_en.update({'alpha':a,'l1':l1,'mae':m_mae})
print(f"ElasticNet best alpha={best_en['alpha']} l1={best_en['l1']} chrono_MAE={best_en['mae']:.4f}")

# Tree model OOF (single parameter set for simplicity)
print('Tree model OOF (chronological)...')
oof_R1 = np.zeros(len(y))  # stable ridge
oof_R2 = np.zeros(len(y))  # full ridge
oof_EN = np.zeros(len(y))  # elastic net
oof_TR = np.zeros(len(y))  # tree

for train_years_concat, val_years in fold_definitions:
    tr_mask = np.isin(train_years, train_years_concat)
    va_mask = np.isin(train_years, val_years)
    if tr_mask.sum()==0 or va_mask.sum()==0:
        continue
    # R1
    r1 = Ridge(alpha=alpha_stable)
    r1.fit(X_stable_s[tr_mask], y.iloc[tr_mask])
    oof_R1[va_mask] = r1.predict(X_stable_s[va_mask])
    # R2
    r2 = Ridge(alpha=alpha_full)
    r2.fit(X_full_s[tr_mask], y.iloc[tr_mask])
    oof_R2[va_mask] = r2.predict(X_full_s[va_mask])
    # EN
    en = ElasticNet(alpha=best_en['alpha'], l1_ratio=best_en['l1'], random_state=RANDOM_STATE, max_iter=5000)
    en.fit(X_en_s[tr_mask], y.iloc[tr_mask])
    oof_EN[va_mask] = en.predict(X_en_s[va_mask])
    # TR
    tr = HistGradientBoostingRegressor(**TREE_PARAMS)
    tr.fit(X_full_s[tr_mask], y.iloc[tr_mask])
    oof_TR[va_mask] = tr.predict(X_full_s[va_mask])

base_oof_stack = np.vstack([oof_R1, oof_R2, oof_EN, oof_TR]).T  # shape (n,4)

# OOF MAEs per base model (chrono)
mae_R1 = mean_absolute_error(y, oof_R1)
mae_R2 = mean_absolute_error(y, oof_R2)
mae_EN = mean_absolute_error(y, oof_EN)
mae_TR = mean_absolute_error(y, oof_TR)
print(f"Base chrono OOF MAEs | R1={mae_R1:.4f} R2={mae_R2:.4f} EN={mae_EN:.4f} TR={mae_TR:.4f}")

# =============================================================================
# META MODEL (Ridge on OOF predictions)
# =============================================================================
print('Training meta ridge (chronological stacking)...')
# Build chronological folds again but on OOF matrix for meta alpha selection
best_meta = {'alpha':None,'mae':np.inf}
for a in META_ALPHA_GRID:
    fold_maes = []
    for train_years_concat, val_years in fold_definitions:
        tr_mask = np.isin(train_years, train_years_concat)
        va_mask = np.isin(train_years, val_years)
        if tr_mask.sum()==0 or va_mask.sum()==0:
            continue
        meta = Ridge(alpha=a)
        meta.fit(base_oof_stack[tr_mask], y.iloc[tr_mask])
        pred = meta.predict(base_oof_stack[va_mask])
        fold_maes.append(mean_absolute_error(y.iloc[va_mask], pred))
    if fold_maes:
        m_mae = np.mean(fold_maes)
        if m_mae < best_meta['mae']:
            best_meta.update({'alpha':a,'mae':m_mae})
print(f"Meta ridge alpha={best_meta['alpha']} chrono_MAE={best_meta['mae']:.4f}")

# Fit final base models on full data
r1_full = Ridge(alpha=alpha_stable).fit(X_stable_s, y)
r2_full = Ridge(alpha=alpha_full).fit(X_full_s, y)
en_full = ElasticNet(alpha=best_en['alpha'], l1_ratio=best_en['l1'], random_state=RANDOM_STATE, max_iter=5000).fit(X_en_s, y)
tr_full = HistGradientBoostingRegressor(**TREE_PARAMS).fit(X_full_s, y)

# Build test stack
test_stack = np.vstack([
    r1_full.predict(X_test_stable_s),
    r2_full.predict(X_test_full_s),
    en_full.predict(X_test_en_s),
    tr_full.predict(X_test_full_s)
]).T

meta_full = Ridge(alpha=best_meta['alpha']).fit(base_oof_stack, y)
final_float = meta_full.predict(test_stack)

# Decade bias correction (optional)
if DECADE_BIAS_CORRECTION and 'yearID' in test_df.columns:
    decades = (train_years//10)*10
    residuals = y - meta_full.predict(base_oof_stack)
    offsets = {}
    for dec in np.unique(decades):
        mask = decades == dec
        if mask.sum()<8: continue
        med = residuals[mask].median()
        adj = float(np.clip(med*DECADE_CORR_SHRINK, -DECADE_CORR_MAX_ABS, DECADE_CORR_MAX_ABS))
        offsets[int(dec)] = adj
    test_decades = (test_df['yearID'].values//10)*10
    dec_offset_vec = np.array([offsets.get(int(d),0.0) for d in test_decades])
    final_float += dec_offset_vec
    decade_offsets = offsets
else:
    decade_offsets = {}

final_pred = np.clip(final_float, 0, 162).round().astype(int)
pd.DataFrame({'ID': test_df['ID'], 'W': final_pred}).to_csv(SUBMISSION_FILE, index=False)
print(f"Saved submission: {SUBMISSION_FILE}")

# Approximate meta weights (standardized) for interpretability
# Refit meta with standardized base features
base_mean = base_oof_stack.mean(axis=0)
base_std = base_oof_stack.std(axis=0) + 1e-8
standardized_oof = (base_oof_stack - base_mean)/base_std
meta_std = Ridge(alpha=best_meta['alpha']).fit(standardized_oof, y)
meta_raw_coefs = meta_std.coef_ / (np.abs(meta_std.coef_).sum()+1e-9)

# Diagnostics
try:
    diag = {
        'alphas': {
            'stable_ridge': alpha_stable,
            'full_ridge': alpha_full,
            'elasticnet_alpha': best_en['alpha'],
            'elasticnet_l1_ratio': best_en['l1'],
            'tree_params': TREE_PARAMS,
            'meta_alpha': best_meta['alpha']
        },
        'chrono_oof_mae_base': {
            'R1_stable': mae_R1,
            'R2_full': mae_R2,
            'EN': mae_EN,
            'TR': mae_TR
        },
        'meta_chrono_mae': best_meta['mae'],
        'meta_weight_proportions': {
            'R1_stable': float(meta_raw_coefs[0]),
            'R2_full': float(meta_raw_coefs[1]),
            'EN': float(meta_raw_coefs[2]),
            'TR': float(meta_raw_coefs[3])
        },
        'decade_offsets': decade_offsets,
        'submission_file': SUBMISSION_FILE,
        'n_features_full': len(all_features),
        'n_features_stable': len(stable_features)
    }
    with open(DIAGNOSTICS_FILE,'w') as f:
        json.dump(diag, f, indent=2)
    print(f"Diagnostics saved: {DIAGNOSTICS_FILE}")
except Exception as e:
    print('Diagnostics save failed:', e)

print('================ TEMPORAL STACK SUMMARY ================')
print(f"Base OOF Chrono MAEs: R1={mae_R1:.4f} R2={mae_R2:.4f} EN={mae_EN:.4f} TR={mae_TR:.4f}")
print(f"Meta Chrono MAE: {best_meta['mae']:.4f}")
print(f"Meta weight proportions (approx): {meta_raw_coefs}")
print(f"Submission: {SUBMISSION_FILE}")
print('========================================================')
