"""
═══════════════════════════════════════════════════════════════════════════════
CHAMPION RECOVERY SCRIPT
═══════════════════════════════════════════════════════════════════════════════
Purpose:
  The V2/V3 experimental expansions increased leaderboard score (regressed to 3.03292).
  This script returns to the strongest core (3 ridge-based models) and adds only
  LOW-RISK stability improvements:
    • Delayed rounding
    • Careful alpha tuning per model
    • Multi-seed for Model 3
    • Robust internal + global blend weight search
    • Bootstrap stability selection (penalizes unstable weight sets)
    • Variance-adjusted scoring: mean_bootstrap_MAE + penalty * std_bootstrap_MAE

Models:
  M1: No-temporal stable ridge
  M2: Two-feature-set internal ridge ensemble (weight tuned by OOF MAE)
  M3: Fine-tuned ridge with multiple seeds

Final blend weights chosen via:
  1. Coarse grid search over (w1,w2,w3)
  2. Select top K by raw OOF MAE
  3. Bootstrap resampling (B iterations) evaluating each candidate
  4. Penalized score = mean_MAE + PENALTY * std_MAE (PENALTY ~ 0.35)
  5. Select min penalized score; fallback to equal weights if needed

Outputs:
  submission_champion_recovery.csv
  blend_diagnostics_recovery.json (weights, MAEs, bootstrap distribution)

Run:
  python generate_champion_recovery.py

Next (if still not improved):
  • Add a single non-linear model (HistGradientBoostingRegressor) for M4
  • Try win% target again gated by bootstrap stability
  • Light residual trimming + refit core models
═══════════════════════════════════════════════════════════════════════════════
Date: 2025-10-07
"""

import numpy as np
import pandas as pd
import json, warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST  = 'data/test.csv'
SUBMISSION_FILE = 'submission_champion_recovery.csv'
DIAGNOSTICS_FILE = 'blend_diagnostics_recovery.json'

N_FOLDS = 10
RANDOM_STATE = 42
BOOTSTRAP_ITER = 400          # Stability evaluation iterations
TOP_K_CANDIDATES = 25          # Number of top raw OOF weight sets to evaluate deeper
PENALTY = 0.35                 # Penalty coefficient for std deviation in bootstrap
MIN_GAIN_REQUIRED = 0.00015    # If candidate not at least this better than equal weights, fallback

# =============================================================================
# CONFIG (extended for optional tree model)
# =============================================================================
TREE_MODEL_ENABLED = True  # Set False to skip Model 4 (HistGradientBoostingRegressor)
FOUR_MODEL_MIN_IMPROVEMENT = 0.001  # required raw OOF MAE improvement over 3-model
TREE_RESIDUAL_CORR_MAX = 0.93       # require residual decorrelation vs strongest model (M3)
FOUR_MODEL_PENALTY = 0.35           # same penalty scheme
FOUR_MODEL_BOOTSTRAP_ITER = 300     # fewer iterations to keep runtime reasonable
FOUR_MODEL_TOP_K = 30               # candidate weight sets to bootstrap

np.random.seed(RANDOM_STATE)

# =============================================================================
# DATA LOADING
# =============================================================================
print('='*90)
print('📊 Loading data (Recovery)')
print('='*90)
train_df = pd.read_csv(DATA_TRAIN)
test_df  = pd.read_csv(DATA_TEST)
y = train_df['W'].astype(float)
print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def stable_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'R','RA','G'}.issubset(df.columns):
        for exp in [1.83,1.85,1.9,2.0]:
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

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# MODEL 1: Stable Ridge (No-temporal)
# =============================================================================
print('\n=== MODEL 1: Stable Ridge ===')
train_m1 = stable_features(train_df)
test_m1  = stable_features(test_df)
feat_m1 = sorted((set(train_m1.columns) & set(test_m1.columns)) - EXCLUDE)
X1 = train_m1[feat_m1].fillna(0).values
T1 = test_m1[feat_m1].fillna(0).values

scaler1 = RobustScaler().fit(X1)
X1s = scaler1.transform(X1)
T1s = scaler1.transform(T1)

alpha_grid_m1 = [0.5,1,2,3,5,7]
best_m1 = {'alpha':None,'mae':np.inf}
for a in alpha_grid_m1:
    scores = cross_val_score(Ridge(alpha=a), X1s, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_m1['mae']:
        best_m1.update({'alpha':a,'mae':mae})
print(f"Model1 best alpha={best_m1['alpha']} CV_MAE={best_m1['mae']:.4f}")

# OOF predictions
oof_m1 = np.zeros(len(y))
for tr,va in kfold.split(X1s):
    m = Ridge(alpha=best_m1['alpha']).fit(X1s[tr], y.iloc[tr])
    oof_m1[va] = m.predict(X1s[va])
model1_full = Ridge(alpha=best_m1['alpha']).fit(X1s, y)
pred1_test_float = model1_full.predict(T1s)
print(f"Model1 OOF MAE: {mean_absolute_error(y,oof_m1):.4f}")

# =============================================================================
# MODEL 2: Two Subset Ridge Ensemble
# =============================================================================
print('\n=== MODEL 2: Feature Subset Ridge Ensemble ===')
# Simple deterministic split by feature name ordering
mid = len(feat_m1)//2
feat_a = feat_m1[:mid]
feat_b = feat_m1[mid:]
Xa = train_m1[feat_a].fillna(0).values
Xb = train_m1[feat_b].fillna(0).values
Ta = test_m1[feat_a].fillna(0).values
Tb = test_m1[feat_b].fillna(0).values

sca = StandardScaler().fit(Xa); scb = StandardScaler().fit(Xb)
Xa_s = sca.transform(Xa); Xb_s = scb.transform(Xb)
Ta_s = sca.transform(Ta); Tb_s = scb.transform(Tb)
alpha_sub = 3.0

# OOF submodels
oof_a = np.zeros(len(y))
oof_b = np.zeros(len(y))
for tr,va in kfold.split(Xa_s):
    ma = Ridge(alpha=alpha_sub).fit(Xa_s[tr], y.iloc[tr])
    mb = Ridge(alpha=alpha_sub).fit(Xb_s[tr], y.iloc[tr])
    oof_a[va] = ma.predict(Xa_s[va])
    oof_b[va] = mb.predict(Xb_s[va])

best_internal = {'w':None,'mae':np.inf}
for w in np.arange(0.30,0.71,0.05):
    blend = w*oof_a + (1-w)*oof_b
    mae = mean_absolute_error(y, blend)
    if mae < best_internal['mae']:
        best_internal.update({'w':w,'mae':mae})
print(f"Internal best weight={best_internal['w']:.2f} OOF_MAE={best_internal['mae']:.4f}")

oof_m2 = best_internal['w']*oof_a + (1-best_internal['w'])*oof_b
ma_full_a = Ridge(alpha=alpha_sub).fit(Xa_s, y)
ma_full_b = Ridge(alpha=alpha_sub).fit(Xb_s, y)
pred2_test_float = best_internal['w']*ma_full_a.predict(Ta_s) + (1-best_internal['w'])*ma_full_b.predict(Tb_s)
print(f"Model2 OOF MAE: {mean_absolute_error(y,oof_m2):.4f}")

# =============================================================================
# MODEL 3: Fine-Tuned Multi-Seed Ridge
# =============================================================================
print('\n=== MODEL 3: Fine-Tuned Multi-Seed Ridge ===')
sc3 = RobustScaler().fit(X1)
X1_sc = sc3.transform(X1); T1_sc = sc3.transform(T1)
alpha_grid_m3 = [0.2,0.25,0.3,0.35,0.4]

best_m3 = {'alpha':None,'mae':np.inf}
for a in alpha_grid_m3:
    scores = cross_val_score(Ridge(alpha=a), X1_sc, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_m3['mae']:
        best_m3.update({'alpha':a,'mae':mae})
print(f"Model3 best alpha={best_m3['alpha']} CV_MAE={best_m3['mae']:.4f}")

# OOF (2 seeds for efficiency)
oof_m3 = np.zeros(len(y))
for tr,va in kfold.split(X1_sc):
    fold_preds = []
    for sd in [42,123]:
        m = Ridge(alpha=best_m3['alpha'], random_state=sd).fit(X1_sc[tr], y.iloc[tr])
        fold_preds.append(m.predict(X1_sc[va]))
    oof_m3[va] = np.mean(fold_preds, axis=0)

# Final test predictions with more seeds
seed_preds = []
for sd in [42,123,456,789,2024]:
    m = Ridge(alpha=best_m3['alpha'], random_state=sd).fit(X1_sc, y)
    seed_preds.append(m.predict(T1_sc))
pred3_test_float = np.mean(seed_preds, axis=0)
print(f"Model3 OOF MAE: {mean_absolute_error(y,oof_m3):.4f}")

# =============================================================================
# OPTIONAL MODEL 4: HistGradientBoostingRegressor (non-linear)
# =============================================================================
M4_INCLUDED = False
try:
    if TREE_MODEL_ENABLED:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print('\n=== MODEL 4: HistGradientBoostingRegressor (candidate) ===')
        # Use same stable feature matrix as Model1 for fairness
        X_tree = X1  # unscaled raw numeric features
        T_tree = T1
        # Basic hyperparameters (keep conservative to avoid overfit)
        hgb_params = dict(max_depth=6, learning_rate=0.06, max_iter=400, l2_regularization=0.6,
                          min_samples_leaf=25, random_state=RANDOM_STATE)
        oof_m4 = np.zeros(len(y))
        for tr, va in kfold.split(X_tree):
            m4 = HistGradientBoostingRegressor(**hgb_params)
            m4.fit(X_tree[tr], y.iloc[tr])
            oof_m4[va] = m4.predict(X_tree[va])
        model4_full = HistGradientBoostingRegressor(**hgb_params)
        model4_full.fit(X_tree, y)
        pred4_test_float = model4_full.predict(T_tree)
        m4_oof_mae = mean_absolute_error(y, oof_m4)
        print(f"Model4 OOF MAE: {m4_oof_mae:.4f}")
        # Residual correlation with strongest ridge (M3)
        m3_resid = y.values - oof_m3
        m4_resid = y.values - oof_m4
        corr_m3_m4 = np.corrcoef(m3_resid, m4_resid)[0,1]
        print(f"Residual corr (M3 vs M4): {corr_m3_m4:.4f}")
        # Decide inclusion: (a) OOF improvement potential AND (b) residual corr below threshold
        best_single_mae = min(mean_absolute_error(y,oof_m1), mean_absolute_error(y,oof_m3))
        include_condition = (m4_oof_mae <= best_single_mae + 0.02) and (corr_m3_m4 <= TREE_RESIDUAL_CORR_MAX)
        if include_condition:
            M4_INCLUDED = True
            print('✓ Including Model 4 in extended blend search.')
        else:
            print('✗ Not including Model 4 (fails criteria).')
    else:
        print('\n(Model 4 disabled by configuration)')
except Exception as e:
    print(f"Model 4 build failed: {e}; skipping.")
    M4_INCLUDED = False

# =============================================================================
# BLENDING: OOF WEIGHT SEARCH + BOOTSTRAP STABILITY
# =============================================================================
print('\n=== GLOBAL BLEND SEARCH (3 Models) ===')
OOF_STACK = np.vstack([oof_m1, oof_m2, oof_m3])
TEST_STACK = np.vstack([pred1_test_float, pred2_test_float, pred3_test_float])

# Generate candidate weights on constrained grid
candidates = []
for w1 in np.arange(0.20,0.61,0.02):
    for w2 in np.arange(0.20,0.61,0.02):
        w3 = 1 - w1 - w2
        if w3 < 0: continue
        if 0.10 <= w3 <= 0.55:
            blend = w1*oof_m1 + w2*oof_m2 + w3*oof_m3
            mae = mean_absolute_error(y, blend)
            candidates.append((mae, (w1,w2,w3)))

candidates.sort(key=lambda x: x[0])
print(f"Total candidate triples: {len(candidates)}")
print("Top 5 raw OOF candidates:")
for m,(w1,w2,w3) in candidates[:5]:
    print(f"  MAE={m:.5f} weights=({w1:.2f},{w2:.2f},{w3:.2f})")

# Equal weight baseline
w_equal = np.array([1/3,1/3,1/3])
mae_equal = mean_absolute_error(y, np.einsum('i,ij->j', w_equal, OOF_STACK))
print(f"Equal weights OOF MAE: {mae_equal:.5f}")

# Bootstrap evaluation of top K candidates
print(f"\nBootstrap evaluating top {TOP_K_CANDIDATES} candidates (B={BOOTSTRAP_ITER}) ...")
indices = np.arange(len(y))
bootstrap_results = []  # list of dicts
TOP = candidates[:TOP_K_CANDIDATES]

# Pre-generate bootstrap samples indices
boot_samples = [np.random.choice(indices, size=len(indices), replace=True) for _ in range(BOOTSTRAP_ITER)]

for raw_mae, w in TOP:
    w_vec = np.array(w)
    base_preds = np.einsum('i,ij->j', w_vec, OOF_STACK)
    maes = []
    for b_idx in boot_samples:
        maes.append(mean_absolute_error(y.iloc[b_idx], base_preds[b_idx]))
    maes = np.array(maes)
    mean_m = maes.mean(); std_m = maes.std()
    penalized = mean_m + PENALTY * std_m
    bootstrap_results.append({
        'weights': w,
        'raw_oof_mae': raw_mae,
        'bootstrap_mean': mean_m,
        'bootstrap_std': std_m,
        'penalized_score': penalized
    })

bootstrap_results.sort(key=lambda d: d['penalized_score'])
print("Top 5 stability-adjusted candidates:")
for d in bootstrap_results[:5]:
    print(f"  pen={d['penalized_score']:.5f} mean={d['bootstrap_mean']:.5f} std={d['bootstrap_std']:.5f} w={d['weights']}")

best_stable = bootstrap_results[0]
w_best = np.array(best_stable['weights'])
print(f"\nChosen stable weights: {w_best} penalized={best_stable['penalized_score']:.5f} raw_oof={best_stable['raw_oof_mae']:.5f}")

# Fallback if not meaningfully better than equal weights
if mae_equal - best_stable['raw_oof_mae'] < MIN_GAIN_REQUIRED:
    print(f"⚠ Improvement over equal weights ({mae_equal - best_stable['raw_oof_mae']:.5f}) < {MIN_GAIN_REQUIRED}; reverting to equal weights.")
    w_best = w_equal.copy()
    final_oof_mae = mae_equal
else:
    final_oof_mae = best_stable['raw_oof_mae']

print(f"Final blend OOF MAE: {final_oof_mae:.5f} using weights {w_best}")

# ================== ADAPTIVE FALLBACK & CALIBRATION EXTENSION ==================
AUTO_DROP_WEAK_MODEL = True
SINGLE_MODEL_FALLBACK_MARGIN = 0.003  # if best single is this much better than blend, use it
TWO_MODEL_GRID_STEP = 0.01
CALIBRATE_BY_DECADE = True
CALIBRATION_MAX_ABS = 1.5  # cap adjustment
ADJUSTED_SUBMISSION_FILE = 'submission_champion_recovery_adjusted.csv'
ADJUSTED_DIAGNOSTICS_FILE = 'blend_diagnostics_recovery_adjusted.json'

# Store single model OOF MAEs
single_oof = {
    'M1': mean_absolute_error(y, oof_m1),
    'M2': mean_absolute_error(y, oof_m2),
    'M3': mean_absolute_error(y, oof_m3)
}
best_single_model = min(single_oof.items(), key=lambda kv: kv[1])
print(f"Best single model: {best_single_model[0]} OOF_MAE={best_single_model[1]:.5f}")

# Condition 1: Single-model fallback
use_single_model = best_single_model[1] + SINGLE_MODEL_FALLBACK_MARGIN < final_oof_mae

# Condition 2: Drop weak Model 2 if much worse than others & re-optimize 2-model blend
weak_gap_threshold = 0.15
drop_model2 = False
if AUTO_DROP_WEAK_MODEL:
    gap_m2 = single_oof['M2'] - min(single_oof['M1'], single_oof['M3'])
    if gap_m2 > weak_gap_threshold:
        drop_model2 = True
        print(f"Auto-dropping Model 2 (gap {gap_m2:.3f} > {weak_gap_threshold})")

# Recompute alternative solutions if needed
chosen_strategy = 'original_blend'
adjusted_weights = w_best.copy()
adjusted_oof_mae = final_oof_mae
adjusted_oof_preds = np.einsum('i,ij->j', adjusted_weights, OOF_STACK)
adjusted_test_float = np.einsum('i,ij->j', adjusted_weights, TEST_STACK)

if drop_model2 and not use_single_model:
    # Optimize 2-model blend between M1 & M3 only
    print('Re-optimizing 2-model blend (M1,M3) ...')
    oof_pair_stack = np.vstack([oof_m1, oof_m3])
    test_pair_stack = np.vstack([pred1_test_float, pred3_test_float])
    best_pair = {'w': None, 'mae': np.inf}
    for w in np.arange(0.05, 0.96, TWO_MODEL_GRID_STEP):
        blend = w * oof_m1 + (1 - w) * oof_m3
        mae = mean_absolute_error(y, blend)
        if mae < best_pair['mae']:
            best_pair.update({'w': w, 'mae': mae})
    print(f"2-model best: w_M1={best_pair['w']:.2f} w_M3={1-best_pair['w']:.2f} OOF_MAE={best_pair['mae']:.5f}")
    if best_pair['mae'] + 1e-9 < adjusted_oof_mae:
        chosen_strategy = 'two_model_blend'
        adjusted_weights = np.array([best_pair['w'], 0.0, 1 - best_pair['w']])  # map into 3-slot structure
        adjusted_oof_mae = best_pair['mae']
        adjusted_oof_preds = best_pair['w'] * oof_m1 + (1 - best_pair['w']) * oof_m3
        adjusted_test_float = best_pair['w'] * pred1_test_float + (1 - best_pair['w']) * pred3_test_float

if use_single_model:
    print('Using single-model fallback (best individual model beats blend).')
    chosen_strategy = 'single_model'
    if best_single_model[0] == 'M1':
        adjusted_weights = np.array([1.0, 0.0, 0.0])
        adjusted_oof_preds = oof_m1
        adjusted_test_float = pred1_test_float
    elif best_single_model[0] == 'M2':
        adjusted_weights = np.array([0.0, 1.0, 0.0])
        adjusted_oof_preds = oof_m2
        adjusted_test_float = pred2_test_float
    else:  # M3
        adjusted_weights = np.array([0.0, 0.0, 1.0])
        adjusted_oof_preds = oof_m3
        adjusted_test_float = pred3_test_float
    adjusted_oof_mae = best_single_model[1]

print(f"Chosen strategy: {chosen_strategy} | Adjusted OOF MAE: {adjusted_oof_mae:.5f}")

# Decade-based calibration (optional)
calibration_offsets = {}
if CALIBRATE_BY_DECADE and 'yearID' in train_df.columns and 'yearID' in test_df.columns:
    decade_train = (train_df['yearID'] // 10) * 10
    base_residuals = y - adjusted_oof_preds
    for dec in sorted(decade_train.unique()):
        dec_mask = decade_train == dec
        if dec_mask.sum() < 5:
            continue
        med = base_residuals[dec_mask].median()
        # shrink small offsets
        shrink = 1 / (1 + np.exp(-abs(med)))  # sigmoid scaling near zero ~0.5
        adj = med * 0.5 * shrink  # partial correction
        adj = float(np.clip(adj, -CALIBRATION_MAX_ABS, CALIBRATION_MAX_ABS))
        calibration_offsets[int(dec)] = adj
    # Apply to test
    decade_test = (test_df['yearID'] // 10) * 10
    adj_vector = decade_test.map(calibration_offsets).fillna(0.0).values
    adjusted_test_float_cal = adjusted_test_float + adj_vector
else:
    adjusted_test_float_cal = adjusted_test_float

# Final adjusted predictions (calibrated) & save
final_adjusted_pred = np.clip(adjusted_test_float_cal, 0, 162).round().astype(int)

pd.DataFrame({'ID': test_df['ID'], 'W': final_adjusted_pred}).to_csv(ADJUSTED_SUBMISSION_FILE, index=False)
print(f"Adjusted submission saved: {ADJUSTED_SUBMISSION_FILE}")

# Save adjusted diagnostics
try:
    adj_diag = {
        'strategy': chosen_strategy,
        'original_blend_oof_mae': float(final_oof_mae),
        'adjusted_oof_mae': float(adjusted_oof_mae),
        'single_model_oof': single_oof,
        'adjusted_weights': adjusted_weights.tolist(),
        'calibration_offsets': calibration_offsets,
        'config': {
            'AUTO_DROP_WEAK_MODEL': AUTO_DROP_WEAK_MODEL,
            'SINGLE_MODEL_FALLBACK_MARGIN': SINGLE_MODEL_FALLBACK_MARGIN,
            'CALIBRATE_BY_DECADE': CALIBRATE_BY_DECADE
        }
    }
    with open(ADJUSTED_DIAGNOSTICS_FILE, 'w') as f:
        json.dump(adj_diag, f, indent=2)
    print(f"Adjusted diagnostics saved: {ADJUSTED_DIAGNOSTICS_FILE}")
except Exception as e:
    print(f"Could not save adjusted diagnostics: {e}")
# ================== END ADAPTIVE EXTENSION ==================

# =============================================================================
# SUMMARY
# =============================================================================
print('\n=== SUMMARY ===')
print(f"Model1 OOF MAE: {mean_absolute_error(y, oof_m1):.5f}")
print(f"Model2 OOF MAE: {mean_absolute_error(y, oof_m2):.5f}")
print(f"Model3 OOF MAE: {mean_absolute_error(y, oof_m3):.5f}")
print(f"Blend OOF MAE: {final_oof_mae:.5f}")
print(f"Weights used: {w_best}")
print('\nNEXT OPTIONS IF NOT IMPROVED:')
print('  1. Add non-linear model (tree-based) + repeat stability selection.')
print('  2. Lower PENALTY to allow more aggressive weights (current 0.35).')
print('  3. Increase BOOTSTRAP_ITER (costly) for more reliable stability scoring.')
print('  4. Introduce residual trimming and refit M1-M3.')
print('  5. Use LightGBM on stable features as M4 and re-run blend pipeline.')
print('\n🚀 Submit this recovery version and compare to previous best (2.97942).')
print('='*90)

# ================== EXTENDED 4-MODEL BLEND (if M4_INCLUDED) ==================
if M4_INCLUDED:
    print('\n=== EXTENDED 4-MODEL BLEND CANDIDATE EVALUATION ===')
    # Build 4-model OOF + Test stacks using adjusted (post-fallback) best strategy output for first 3
    # Use original per-model OOF (not adjusted calibrations) for fair comparison
    OOF_STACK_4 = np.vstack([oof_m1, oof_m2, oof_m3, oof_m4])
    TEST_STACK_4 = np.vstack([pred1_test_float, pred2_test_float, pred3_test_float, pred4_test_float])

    # Generate candidate weights: coarse grid (step 0.05) controlling maximum enumeration
    four_candidates = []
    steps = np.arange(0.05, 0.91, 0.05)
    for w1 in steps:
        for w2 in steps:
            for w3 in steps:
                total = w1 + w2 + w3
                if total >= 0.95:  # leave room for w4
                    continue
                w4 = 1 - total
                if w4 < 0.05 or w4 > 0.70:
                    continue
                # quick sanity: each weight <=0.80
                if max(w1,w2,w3,w4) > 0.80:
                    continue
                blend = w1*oof_m1 + w2*oof_m2 + w3*oof_m3 + w4*oof_m4
                mae = mean_absolute_error(y, blend)
                four_candidates.append((mae, (w1,w2,w3,w4)))
    four_candidates.sort(key=lambda x: x[0])
    print(f"Generated {len(four_candidates)} 4-model candidates. Top 5 raw:")
    for m,(a,b,c,d) in four_candidates[:5]:
        print(f"  MAE={m:.5f} w=({a:.2f},{b:.2f},{c:.2f},{d:.2f})")

    # If no raw improvement skip
    best_four_raw = four_candidates[0][0] if four_candidates else 1e9
    if best_four_raw + 1e-9 < final_oof_mae - FOUR_MODEL_MIN_IMPROVEMENT:
        print(f"Proceeding to 4-model bootstrap (raw best {best_four_raw:.5f} < 3-model {final_oof_mae:.5f})")
        top_four = four_candidates[:FOUR_MODEL_TOP_K]
        idx_all = np.arange(len(y))
        boot_indices = [np.random.choice(idx_all, size=len(idx_all), replace=True) for _ in range(FOUR_MODEL_BOOTSTRAP_ITER)]
        four_bootstrap = []
        for raw_mae, w in top_four:
            w_vec = np.array(w)
            base_preds = np.einsum('i,ij->j', w_vec, OOF_STACK_4)
            maes = []
            for bi in boot_indices:
                maes.append(mean_absolute_error(y.iloc[bi], base_preds[bi]))
            maes = np.array(maes)
            mean_m = maes.mean(); std_m = maes.std(); penal = mean_m + FOUR_MODEL_PENALTY * std_m
            four_bootstrap.append({
                'weights': w,
                'raw_oof_mae': raw_mae,
                'bootstrap_mean': mean_m,
                'bootstrap_std': std_m,
                'penalized_score': penal
            })
        four_bootstrap.sort(key=lambda d: d['penalized_score'])
        print('Top 5 4-model penalized candidates:')
        for d in four_bootstrap[:5]:
            print(f"  pen={d['penalized_score']:.5f} mean={d['bootstrap_mean']:.5f} std={d['bootstrap_std']:.5f} w={d['weights']}")
        best_four = four_bootstrap[0]
        # Accept only if penalized candidate meaningfully better raw MAE
        if best_four['raw_oof_mae'] + 1e-9 < final_oof_mae - FOUR_MODEL_MIN_IMPROVEMENT:
            print(f"✓ Adopting 4-model blend (raw {best_four['raw_oof_mae']:.5f} < {final_oof_mae:.5f})")
            w_four = np.array(best_four['weights'])
            final_four_test_float = np.einsum('i,ij->j', w_four, TEST_STACK_4)
            final_four_pred = np.clip(final_four_test_float, 0, 162).round().astype(int)
            four_sub_file = 'submission_champion_recovery_tree.csv'
            pd.DataFrame({'ID': test_df['ID'], 'W': final_four_pred}).to_csv(four_sub_file, index=False)
            print(f"Saved 4-model submission: {four_sub_file}")
            # Save diagnostics
            tree_diag = {
                'base_3_model_oof': float(final_oof_mae),
                'four_model_best_raw_oof': float(best_four['raw_oof_mae']),
                'four_model_best_penalized': float(best_four['penalized_score']),
                'four_model_weights': best_four['weights'],
                'tree_residual_corr_M3': float(corr_m3_m4),
                'tree_model_oof_mae': float(m4_oof_mae)
            }
            with open('blend_diagnostics_recovery_tree.json','w') as f:
                json.dump(tree_diag, f, indent=2)
            print('4-model diagnostics saved: blend_diagnostics_recovery_tree.json')
        else:
            print('✗ 4-model penalized improvement insufficient; retaining previous solution.')
    else:
        print('4-model raw candidates do not beat 3-model blend by required margin; skipping.')
