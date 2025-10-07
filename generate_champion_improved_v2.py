"""
═══════════════════════════════════════════════════════════════════════════════
🏆 IMPROVED CHAMPION V2 (Adds Win% Model + Advanced Blend Search) 🏆
═══════════════════════════════════════════════════════════════════════════════
Changes vs `generate_champion_improved.py` (score observed: 2.97942):
1. Adds MODEL 4 (Win Percentage Target): model W_pct = W / G to stabilize variance, then invert.
2. Adds richer interaction & ratio features for Model 4 (OPS, WHIP, K_per_9, run diff ratios, OBP*SLG, etc.).
3. Advanced Weight Optimization: Replaces pure grid search with hybrid approach:
   • Coarse random Dirichlet sampling (reproducible) over 4-way weights (Models 1–4).
   • Local neighborhood refinement around the best sample with decreasing radius.
   • Non-negativity + sum-to-one enforced.
4. Optional fine local micro-refinement with deterministic edge exploration.
5. Single final rounding still applied once at very end.

Goal: Achieve MAE improvement below previous 2.97942 public LB (subject to leaderboard noise).

Future (not yet included):
   - L1/L2 diversified model family (Lasso/ElasticNet) for residual decorrelation.
   - Coordinate descent on MAE using subgradient sign analysis.
   - Residual trimming / robust re-fit.

Run: python generate_champion_improved_v2.py
Output: submission_champion_improved_v2.csv
Date: 2025-10-07
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST = 'data/test.csv'
OUTPUT_FILE = 'submission_champion_improved_v2.csv'
N_FOLDS = 10
RANDOM_STATE = 42

# Random search config for blend weights
DIRICHLET_SAMPLES = 4000   # Coarse exploration
LOCAL_REFINEMENT_ROUNDS = 4
NEIGHBORHOOD_SAMPLES = 500  # per refinement round
INITIAL_RADIUS = 0.20

# CONFIG additions
ENABLE_MODEL4 = True  # Set False to force using only first 3 models quickly
FOUR_MODEL_IMPROVEMENT_MIN_GAIN = 0.0005  # Require at least this OOF MAE gain vs best 3-model
THREE_MODEL_PENALTY_TOL = 0.0002  # If 4-model worse by more than this, drop Model4
DIAGNOSTICS_JSON = 'blend_diagnostics_v2.json'

np.random.seed(RANDOM_STATE)

# ============================================================================
# DATA
# ============================================================================
print('='*90)
print('📊 Loading data (V2)')
print('='*90)
train_df = pd.read_csv(DATA_TRAIN)
test_df = pd.read_csv(DATA_TEST)
y = train_df['W'].astype(float)
print(f'Train: {train_df.shape} | Test: {test_df.shape}')
print()

# ============================================================================
# FEATURE GENERATION (reuse from prior + extended interactions)
# ============================================================================

def stable_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum():
            df[col] = df[col].fillna(df[col].median())
    return df

def feature_set_1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'R','RA','G'}.issubset(df.columns):
        for exp in [1.83, 2.0]:
            exp_str = str(int(exp*100))
            df[f'pyth_wins_{exp_str}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp + 1)) * df['G']
        df['run_diff_per_game'] = (df['R'] - df['RA']) / (df['G'] + 1)
    return df

def feature_set_2(df: pd.DataFrame) -> pd.DataFrame:
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

def model4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Richer interaction/ratio features for Win% model."""
    df = stable_features(df)
    # Interaction terms (kept small to avoid overfitting ridge)
    inters = []
    if {'OBP','SLG'}.issubset(df.columns):
        df['OBP_times_SLG'] = df['OBP'] * df['SLG']; inters.append('OBP_times_SLG')
    if {'run_diff_per_game','OBP'}.issubset(df.columns):
        df['rdpg_times_OBP'] = df['run_diff_per_game'] * df['OBP']; inters.append('rdpg_times_OBP')
    if {'K_per_9','WHIP'}.issubset(df.columns):
        df['K9_div_WHIP'] = df['K_per_9'] / (df['WHIP'] + 0.01); inters.append('K9_div_WHIP')
    if {'run_ratio','WHIP'}.issubset(df.columns):
        df['runratio_div_WHIP'] = df['run_ratio'] / (df['WHIP'] + 0.01); inters.append('runratio_div_WHIP')
    # Log transforms (stabilize heavy tails)
    for col in ['R','RA','HR','SO','BB']:
        if col in df.columns:
            df[f'log1p_{col}'] = np.log1p(df[col])
    # Clean again
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

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ============================================================================
# MODEL 1: No-Temporal (reuse logic from improved v1)
# ============================================================================
print('='*90)
print('MODEL 1: No-Temporal Enhanced')
print('='*90)
train_m1 = stable_features(train_df)
test_m1 = stable_features(test_df)
feat_m1 = sorted((set(train_m1.columns) & set(test_m1.columns)) - EXCLUDE_BASE)
X1_train = train_m1[feat_m1].fillna(0); X1_test = test_m1[feat_m1].fillna(0)
scalers_m1 = [StandardScaler(), RobustScaler()]
alpha_grid_m1 = [0.5,1,2,3,5,7,10]

best_m1 = {'mae': np.inf, 'scaler': None, 'alpha': None}
for sc in scalers_m1:
    X_scaled = sc.fit_transform(X1_train)
    for a in alpha_grid_m1:
        ridge = Ridge(alpha=a)
        scores = cross_val_score(ridge, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -scores.mean()
        if mae < best_m1['mae']:
            best_m1.update({'mae': mae, 'scaler': sc, 'alpha': a})
print(f"Best M1 alpha={best_m1['alpha']} scaler={best_m1['scaler'].__class__.__name__} CV_MAE={best_m1['mae']:.4f}")

X1_train_scaled = best_m1['scaler'].fit_transform(X1_train)
X1_test_scaled = best_m1['scaler'].transform(X1_test)
model1 = Ridge(alpha=best_m1['alpha']).fit(X1_train_scaled, y)
pred_m1_test = model1.predict(X1_test_scaled)

# OOF
oof_m1 = np.zeros(len(y))
for tr, va in kfold.split(X1_train_scaled):
    m = Ridge(alpha=best_m1['alpha'])
    m.fit(X1_train_scaled[tr], y.iloc[tr])
    oof_m1[va] = m.predict(X1_train_scaled[va])
print(f"Model 1 OOF MAE: {mean_absolute_error(y, oof_m1):.4f}\n")

# ============================================================================
# MODEL 2: Multi-Ensemble (same as improved v1 with OOF internal weight search)
# ============================================================================
print('='*90)
print('MODEL 2: Multi-Ensemble With Internal Weight Optimization')
print('='*90)
train_s1 = feature_set_1(train_df); test_s1 = feature_set_1(test_df)
train_s2 = feature_set_2(train_df); test_s2 = feature_set_2(test_df)
feat_s1 = sorted((set(train_s1.columns) & set(test_s1.columns)) - EXCLUDE_MULTI)
feat_s2 = sorted((set(train_s2.columns) & set(test_s2.columns)) - EXCLUDE_MULTI)
X2a_train = train_s1[feat_s1].fillna(0); X2b_train = train_s2[feat_s2].fillna(0)
X2a_test = test_s1[feat_s1].fillna(0);  X2b_test = test_s2[feat_s2].fillna(0)
scaler2a = StandardScaler().fit(X2a_train); scaler2b = StandardScaler().fit(X2b_train)
X2a_train_scaled = scaler2a.transform(X2a_train); X2b_train_scaled = scaler2b.transform(X2b_train)
X2a_test_scaled = scaler2a.transform(X2a_test); X2b_test_scaled = scaler2b.transform(X2b_test)
alpha_sub = 3.0

# OOF submodels
oof_sub1 = np.zeros(len(y)); oof_sub2 = np.zeros(len(y))
for tr, va in kfold.split(X2a_train_scaled):
    m_a = Ridge(alpha=alpha_sub); m_a.fit(X2a_train_scaled[tr], y.iloc[tr]); oof_sub1[va] = m_a.predict(X2a_train_scaled[va])
    m_b = Ridge(alpha=alpha_sub); m_b.fit(X2b_train_scaled[tr], y.iloc[tr]); oof_sub2[va] = m_b.predict(X2b_train_scaled[va])

best_internal = {'w1': None, 'mae': np.inf}
for w1 in np.arange(0.30,0.71,0.05):
    blend = w1*oof_sub1 + (1-w1)*oof_sub2
    mae = mean_absolute_error(y, blend)
    if mae < best_internal['mae']:
        best_internal.update({'w1': w1, 'mae': mae})
print(f"Internal weights: sub1={best_internal['w1']:.2f} sub2={1-best_internal['w1']:.2f} OOF_MAE={best_internal['mae']:.4f}")

model2_sub1 = Ridge(alpha=alpha_sub).fit(X2a_train_scaled, y)
model2_sub2 = Ridge(alpha=alpha_sub).fit(X2b_train_scaled, y)
sub1_test_pred = model2_sub1.predict(X2a_test_scaled)
sub2_test_pred = model2_sub2.predict(X2b_test_scaled)
pred_m2_test = best_internal['w1']*sub1_test_pred + (1-best_internal['w1'])*sub2_test_pred

oof_m2 = best_internal['w1']*oof_sub1 + (1-best_internal['w1'])*oof_sub2
print(f"Model 2 OOF MAE: {mean_absolute_error(y, oof_m2):.4f}\n")

# ============================================================================
# MODEL 3: Fine-Tuned Multi-Seed Ridge
# ============================================================================
print('='*90)
print('MODEL 3: Fine-Tuned Ridge (Alpha micro-grid + seeds)')
print('='*90)
train_ft = stable_features(train_df); test_ft = stable_features(test_df)
feat_ft = sorted((set(train_ft.columns) & set(test_ft.columns)) - EXCLUDE_BASE)
X3_train = train_ft[feat_ft].fillna(0); X3_test = test_ft[feat_ft].fillna(0)
scaler3 = RobustScaler().fit(X3_train)
X3_train_scaled = scaler3.transform(X3_train); X3_test_scaled = scaler3.transform(X3_test)
alpha_grid_ft = [0.2,0.25,0.3,0.35,0.4,0.45,0.5]

best_ft = {'alpha': None, 'mae': np.inf}
for a in alpha_grid_ft:
    ridge = Ridge(alpha=a, random_state=RANDOM_STATE)
    scores = cross_val_score(ridge, X3_train_scaled, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_ft['mae']:
        best_ft.update({'alpha': a, 'mae': mae})
print(f"Best M3 alpha={best_ft['alpha']} CV_MAE={best_ft['mae']:.4f}")

seeds_m3 = [42,123,456,789,2024]
seed_preds = []
for sd in seeds_m3:
    m = Ridge(alpha=best_ft['alpha'], random_state=sd)
    m.fit(X3_train_scaled, y)
    seed_preds.append(m.predict(X3_test_scaled))
pred_m3_test = np.mean(seed_preds, axis=0)

oof_m3 = np.zeros(len(y))
for tr, va in kfold.split(X3_train_scaled):
    fold_preds = []
    for sd in [42,123]:
        m = Ridge(alpha=best_ft['alpha'], random_state=sd)
        m.fit(X3_train_scaled[tr], y.iloc[tr])
        fold_preds.append(m.predict(X3_train_scaled[va]))
    oof_m3[va] = np.mean(fold_preds, axis=0)
print(f"Model 3 OOF MAE: {mean_absolute_error(y, oof_m3):.4f}\n")

# ============================================================================
# MODEL 4: Win% Target (NEW)
# ============================================================================
print('='*90)
print('MODEL 4: Win% Target Transformation (W/G) + Interactions')
print('='*90)
train_m4 = model4_features(train_df)
test_m4  = model4_features(test_df)

if 'G' not in train_m4.columns:
    raise ValueError('Column G required for win percentage modeling.')

# Target transform
G_train = train_m4['G'].astype(float)
y_pct = y / (G_train + 1e-6)

feat_m4 = sorted((set(train_m4.columns) & set(test_m4.columns)) - EXCLUDE_BASE)
X4_train = train_m4[feat_m4].fillna(0)
X4_test  = test_m4[feat_m4].fillna(0)

scaler4 = StandardScaler().fit(X4_train)
X4_train_scaled = scaler4.transform(X4_train)
X4_test_scaled  = scaler4.transform(X4_test)

alpha_grid_m4 = [0.05,0.1,0.2,0.3,0.4,0.5]

best_m4 = {'alpha': None, 'mae': np.inf}
for a in alpha_grid_m4:
    ridge = Ridge(alpha=a)
    scores = cross_val_score(ridge, X4_train_scaled, y_pct, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_m4['mae']:
        best_m4.update({'alpha': a, 'mae': mae})
print(f"Best M4 alpha={best_m4['alpha']} (Win% CV MAE={best_m4['mae']:.5f})")

model4 = Ridge(alpha=best_m4['alpha']).fit(X4_train_scaled, y_pct)
G_test = test_m4['G'].astype(float)
pred_pct_test = model4.predict(X4_test_scaled)
pred_m4_test = pred_pct_test * G_test  # invert

# OOF predictions for Model 4
oof_m4 = np.zeros(len(y))
for tr, va in kfold.split(X4_train_scaled):
    m = Ridge(alpha=best_m4['alpha'])
    m.fit(X4_train_scaled[tr], y_pct.iloc[tr])
    oof_pct = m.predict(X4_train_scaled[va])
    oof_m4[va] = oof_pct * G_train.iloc[va]
print(f"Model 4 OOF MAE (inverted W): {mean_absolute_error(y, oof_m4):.4f}\n")

# ============================================================================
# BLEND OPTIMIZATION (4 models with fallback to 3) : Random Dirichlet + Refinement
# ============================================================================
print('='*90)
print('🔧 Global blend weight optimization with fallback strategy')
print('='*90)

OOF_STACK = np.vstack([oof_m1, oof_m2, oof_m3])  # start with 3 models
TEST_STACK = np.vstack([pred_m1_test, pred_m2_test, pred_m3_test])
model_names = ['M1','M2','M3']

if ENABLE_MODEL4:
    OOF_STACK_4 = np.vstack([oof_m1, oof_m2, oof_m3, oof_m4])
    TEST_STACK_4 = np.vstack([pred_m1_test, pred_m2_test, pred_m3_test, pred_m4_test])
else:
    OOF_STACK_4 = OOF_STACK
    TEST_STACK_4 = TEST_STACK

# Helper to optimize weights via random Dirichlet + local refinement

def optimize_weights(oof_matrix, random_samples=3000, seeds=(42,), local_rounds=3, local_samples=300, init_radius=0.15):
    rng = np.random.default_rng(seeds[0])
    k, n = oof_matrix.shape
    # baseline equal weights
    base_w = np.ones(k) / k
    best = {
        'weights': base_w.copy(),
        'mae': mean_absolute_error(y, np.einsum('i,ij->j', base_w, oof_matrix))
    }
    dirichlet = rng.dirichlet(np.ones(k), size=random_samples)
    for w in dirichlet:
        pred = np.einsum('i,ij->j', w, oof_matrix)
        mae = mean_absolute_error(y, pred)
        if mae < best['mae']:
            best['mae'] = mae
            best['weights'] = w.copy()
    # local refinement
    w0 = best['weights'].copy()
    for r in range(local_rounds):
        rad = init_radius * (0.5 ** r)
        for _ in range(local_samples):
            noise = rng.normal(0, rad, size=k)
            w_c = w0 + noise
            w_c = np.clip(w_c, 0, None)
            if w_c.sum() == 0:
                continue
            w_c /= w_c.sum()
            pred = np.einsum('i,ij->j', w_c, oof_matrix)
            mae = mean_absolute_error(y, pred)
            if mae < best['mae']:
                best['mae'] = mae
                best['weights'] = w_c.copy()
                w0 = w_c.copy()
    # micro edge tweaks
    base = best['weights'].copy()
    for i in range(k):
        for delta in [0.01, -0.01, 0.005, -0.005]:
            w_c = base.copy()
            w_c[i] += delta
            if np.any(w_c < 0):
                continue
            w_c /= w_c.sum()
            pred = np.einsum('i,ij->j', w_c, oof_matrix)
            mae = mean_absolute_error(y, pred)
            if mae < best['mae']:
                best['mae'] = mae
                best['weights'] = w_c.copy()
    return best

# Optimize 3-model baseline
print('Optimizing 3-model blend (M1,M2,M3)...')
result_3 = optimize_weights(OOF_STACK, random_samples=2500, local_rounds=3, local_samples=400)
print(f"3-model best MAE: {result_3['mae']:.5f} weights={result_3['weights']}")

if ENABLE_MODEL4:
    print('Optimizing 4-model blend (M1,M2,M3,M4)...')
    result_4 = optimize_weights(OOF_STACK_4, random_samples=4000, local_rounds=4, local_samples=500)
    print(f"4-model best MAE: {result_4['mae']:.5f} weights={result_4['weights']}")

    # Decision logic
    improvement = result_3['mae'] - result_4['mae']
    if improvement >= FOUR_MODEL_IMPROVEMENT_MIN_GAIN:
        chosen = result_4
        chosen_stack = TEST_STACK_4
        chosen_names = ['M1','M2','M3','M4']
        print(f"✓ Using 4-model blend (improvement {improvement:.5f} >= {FOUR_MODEL_IMPROVEMENT_MIN_GAIN})")
    elif improvement < -THREE_MODEL_PENALTY_TOL:
        chosen = result_3
        chosen_stack = TEST_STACK
        chosen_names = ['M1','M2','M3']
        print(f"⚠ 4-model performed worse by {-improvement:.5f}; reverting to 3-model blend")
    else:
        # marginal improvement: shrink towards 3-model weights to reduce variance
        print(f"~ Marginal 4-model improvement ({improvement:.5f}); applying shrinkage blend")
        w3_ext = np.concatenate([result_3['weights'], [0]]) if OOF_STACK_4.shape[0] == 4 else result_3['weights']
        shrink_factor = 0.5  # halfway between
        w_final = shrink_factor * result_4['weights'] + (1 - shrink_factor) * w3_ext
        w_final /= w_final.sum()
        mae_shrunk = mean_absolute_error(y, np.einsum('i,ij->j', w_final, OOF_STACK_4))
        chosen = {'weights': w_final, 'mae': mae_shrunk}
        chosen_stack = TEST_STACK_4
        chosen_names = ['M1','M2','M3','M4']
        print(f"Shrunk weights MAE: {mae_shrunk:.5f} weights={w_final}")
else:
    chosen = result_3
    chosen_stack = TEST_STACK
    chosen_names = ['M1','M2','M3']

print(f"Final chosen blend MAE: {chosen['mae']:.5f} using models {chosen_names}")

# Residual correlation diagnostics
print('\nResidual correlation matrix (chosen models):')
indices = [0,1,2] if not ENABLE_MODEL4 or (ENABLE_MODEL4 and len(chosen['weights'])==3) else [0,1,2,3]
residuals = []
for idx in indices:
    oof_pred = OOF_STACK_4[idx] if ENABLE_MODEL4 else OOF_STACK[idx]
    residuals.append(y.values - oof_pred)
res_mat = np.corrcoef(residuals)
for i,row in enumerate(res_mat):
    label_i = chosen_names[i]
    row_fmt = ' '.join(f"{v:6.3f}" for v in row)
    print(f"  {label_i}: {row_fmt}")

# Save diagnostics to JSON
try:
    diagnostics = {
        'chosen_models': chosen_names,
        'chosen_weights': chosen['weights'].tolist(),
        'chosen_oof_mae': chosen['mae'],
        'three_model_mae': result_3['mae'],
        'four_model_mae': (result_4['mae'] if ENABLE_MODEL4 else None),
        'residual_corr_matrix': res_mat.tolist()
    }
    with open(DIAGNOSTICS_JSON, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Diagnostics saved to {DIAGNOSTICS_JSON}")
except Exception as e:
    print(f"Could not save diagnostics: {e}")

# Store for final prediction phase
final_weight_vector = chosen['weights']
FINAL_TEST_STACK = chosen_stack

# ============================================================================
# FINAL PREDICTIONS (updated to use fallback-selected blend)
# ============================================================================
print('='*90)
print('🏁 Generating final predictions & submission (single rounding)')
print('='*90)
final_float = np.einsum('i,ij->j', final_weight_vector, FINAL_TEST_STACK)
final_pred = np.clip(final_float, 0, 162).round().astype(int)

submission = pd.DataFrame({'ID': test_df['ID'], 'W': final_pred})
submission.to_csv(OUTPUT_FILE, index=False)

print('Prediction distribution:')
print(f"  Min={final_pred.min()} Max={final_pred.max()} Mean={final_pred.mean():.2f} Std={final_pred.std():.2f}")
print(f"Saved: {OUTPUT_FILE} (rows={len(submission)})")
print('\nModel OOF MAEs:')
print(f"  M1: {mean_absolute_error(y, oof_m1):.5f}")
print(f"  M2: {mean_absolute_error(y, oof_m2):.5f}")
print(f"  M3: {mean_absolute_error(y, oof_m3):.5f}")
print(f"  M4: {mean_absolute_error(y, oof_m4):.5f}")
print(f"Blend (chosen) OOF MAE: {chosen['mae']:.5f}")
print('Chosen weights / models:')
for n, w in zip(chosen_names, final_weight_vector):
    print(f"  {n}: {w:.4f}")
print('\nNotes:')
print(' - If improvement is marginal, next step: add ElasticNet & Lasso variants for diversity.')
print(' - Could also try coordinate descent on MAE with soft sign updates; contact if desired.')
print(' - Consider adding residual trimming (remove top 0.5% abs residual rows) then refit M1/M2.')
print('\n🚀 Evaluate on Kaggle and compare vs v1 (2.97942).')
print('='*90)
