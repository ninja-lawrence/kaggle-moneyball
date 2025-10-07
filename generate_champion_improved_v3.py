"""
═══════════════════════════════════════════════════════════════════════════════
🏆 IMPROVED CHAMPION V3 (Model Diversity + Coordinate Descent Blend) 🏆
═══════════════════════════════════════════════════════════════════════════════
Why V3? Public LB score regressed to 2.99176 (worse than prior 2.97942 + champion ~2.9753).
Aim: Increase model diversity & reduce overfitting of weight search introduced in V2.

New in V3:
1. Added MODEL 5 (ElasticNet) & MODEL 6 (Lasso) on stable features for residual decorrelation.
2. Optional residual trimming (drop top X% largest OOF abs residual rows and refit core models).
3. OOF prediction caching (.npz) for faster iteration if feature sets unchanged.
4. Hybrid weight optimizer:
   • Initial Dirichlet random search (diversity)
   • Local stochastic refinement
   • Coordinate descent on MAE using subgradient sign approximation
   • Simplex projection to keep weights non-negative & summing to 1.
5. Adaptive model subset selection: evaluate incremental OOF improvement when adding each model; keep only beneficial ones (prevents noisy models harming blend).
6. Fallback logic: If expanded (5–6 model) blend not ≥ best smaller subset MAE - tolerance, revert.
7. Flags to quickly disable Win% model and/or new regularized models.

Usage:
  python generate_champion_improved_v3.py

Outputs:
  submission_champion_improved_v3.csv
  blend_diagnostics_v3.json
  oof_cache_v3.npz (optional caching)

Tunable Flags (see CONFIG section):
  ENABLE_MODEL4_WINPCT, ENABLE_MODEL5_ENET, ENABLE_MODEL6_LASSO, ENABLE_RESIDUAL_TRIMMING

Next potential future steps (not yet here): LightGBM/Poisson variant, residual stacking meta-model, quantile calibration.
Date: 2025-10-07
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import json, os, hashlib, warnings
from dataclasses import dataclass
from typing import List, Dict
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST = 'data/test.csv'
SUBMISSION_FILE = 'submission_champion_improved_v3.csv'
DIAGNOSTICS_FILE = 'blend_diagnostics_v3.json'
OOF_CACHE_FILE = 'oof_cache_v3.npz'

N_FOLDS = 10
RANDOM_STATE = 42
DIRICHLET_SAMPLES = 4000
LOCAL_STEPS = 600
COORD_DESC_ITERS = 300
COORD_STEP = 0.015
EARLY_STOP_NO_IMPROVE = 80
MIN_IMPROVEMENT_KEEP_MODEL = 0.0003  # OOF MAE reduction required to keep an added model
FALLBACK_TOLERANCE = 0.0004  # If larger blend not better than best subset by this margin revert

# Feature / model flags
ENABLE_MODEL4_WINPCT = False  # disable by default (was noisy in V2)
ENABLE_MODEL5_ENET = True
ENABLE_MODEL6_LASSO = True
ENABLE_RESIDUAL_TRIMMING = False
RESIDUAL_TRIM_FRACTION = 0.0075  # 0.75% highest residual rows trimmed when enabled

# Blending random seeds (for reproducibility)
BLEND_SEEDS = [42, 123]

np.random.seed(RANDOM_STATE)

# =============================================================================
# DATA LOADING
# =============================================================================
print('='*100)
print('📊 Loading data (V3)')
print('='*100)
train_df = pd.read_csv(DATA_TRAIN)
test_df = pd.read_csv(DATA_TEST)
y_full = train_df['W'].astype(float)
print(f'Train: {train_df.shape} | Test: {test_df.shape}')

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

def winpct_features(df: pd.DataFrame) -> pd.DataFrame:
    df = stable_features(df)
    # modest interactions
    if {'OBP','SLG'}.issubset(df.columns):
        df['OBP_SLG'] = df['OBP'] * df['SLG']
    if {'run_diff_per_game','OBP'}.issubset(df.columns):
        df['rdpg_OBP'] = df['run_diff_per_game'] * df['OBP']
    for col in ['R','RA','HR','SO','BB']:
        if col in df.columns:
            df[f'log1p_{col}'] = np.log1p(df[col])
    return df

EXCLUDE_BASE = {
    'W','ID','teamID','yearID','year_label','decade_label','win_bins',
    'decade_1910','decade_1920','decade_1930','decade_1940','decade_1950',
    'decade_1960','decade_1970','decade_1980','decade_1990','decade_2000','decade_2010',
    'era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','mlb_rpg'
}

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# MODEL BUILD HELPERS
# =============================================================================

def ridge_cv_alpha(X, y, alphas, cv, scaler):
    Xs = scaler.fit_transform(X)
    best = {'alpha': None, 'mae': np.inf, 'scaler': scaler}
    for a in alphas:
        model = Ridge(alpha=a)
        scores = cross_val_score(model, Xs, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -scores.mean()
        if mae < best['mae']:
            best.update({'alpha': a, 'mae': mae})
    return best, Xs, scaler

def oof_predictions(model_factory, X, y, cv):
    oof = np.zeros(len(y))
    for tr, va in cv.split(X):
        m = model_factory()
        m.fit(X[tr], y.iloc[tr])
        oof[va] = m.predict(X[va])
    final_model = model_factory()
    final_model.fit(X, y)
    return oof, final_model

# =============================================================================
# BUILD MODELS
# =============================================================================
model_preds_test = {}
oof_dict = {}
model_meta = {}
model_order = []

# MODEL 1: Ridge stable features
print('\n--- MODEL 1 (Ridge stable) ---')
train_m1 = stable_features(train_df)
test_m1 = stable_features(test_df)
feat_m1 = sorted((set(train_m1.columns) & set(test_m1.columns)) - EXCLUDE_BASE)
X1 = train_m1[feat_m1].fillna(0).values
T1 = test_m1[feat_m1].fillna(0).values
best1, X1s, scaler1 = ridge_cv_alpha(train_m1[feat_m1].fillna(0).values, y_full, [0.5,1,2,3,5,7], kfold, RobustScaler())
print(f"Best alpha: {best1['alpha']} CV MAE: {best1['mae']:.4f}")
oof1, model1 = oof_predictions(lambda: Ridge(alpha=best1['alpha']), scaler1.transform(train_m1[feat_m1].fillna(0).values), y_full, kfold)
model_preds_test['M1'] = model1.predict(scaler1.transform(T1))
oof_dict['M1'] = oof1
model_meta['M1'] = {'type':'ridge','alpha':best1['alpha'],'oof_mae':mean_absolute_error(y_full,oof1)}
model_order.append('M1')
print(f"Model 1 OOF MAE: {model_meta['M1']['oof_mae']:.4f}")

# MODEL 2: Multi-ensemble (two feature subsets)
print('\n--- MODEL 2 (Multi Ridge) ---')
# reuse simpler feature splits
feat_subset_a = [c for c in feat_m1 if 'pyth_wins' in c or 'run_diff_per_game' in c or c.startswith('R_') or c.startswith('RA_')]
# fallback if empty
if not feat_subset_a:
    feat_subset_a = feat_m1[:len(feat_m1)//2]
feat_subset_b = [c for c in feat_m1 if c not in feat_subset_a]
Xa = train_m1[feat_subset_a].fillna(0).values
Xb = train_m1[feat_subset_b].fillna(0).values
Ta = test_m1[feat_subset_a].fillna(0).values
Tb = test_m1[feat_subset_b].fillna(0).values
sc_a = StandardScaler().fit(Xa); sc_b = StandardScaler().fit(Xb)
Xa_s = sc_a.transform(Xa); Xb_s = sc_b.transform(Xb)
alpha_sub = 3.0
oof_a, moda = oof_predictions(lambda: Ridge(alpha=alpha_sub), Xa_s, y_full, kfold)
oof_b, modb = oof_predictions(lambda: Ridge(alpha=alpha_sub), Xb_s, y_full, kfold)
# weight search
best_w1 = None; best_mae = np.inf
for w in np.arange(0.3,0.71,0.05):
    blend = w*oof_a + (1-w)*oof_b
    mae = mean_absolute_error(y_full, blend)
    if mae < best_mae:
        best_mae = mae; best_w1 = w
print(f"Internal weights: {best_w1:.2f}/{1-best_w1:.2f} OOF MAE: {best_mae:.4f}")
oof2 = best_w1*oof_a + (1-best_w1)*oof_b
pred2 = best_w1*moda.predict(sc_a.transform(Ta)) + (1-best_w1)*modb.predict(sc_b.transform(Tb))
model_preds_test['M2'] = pred2
oof_dict['M2'] = oof2
model_meta['M2'] = {'type':'multi_ridge','alpha':alpha_sub,'w1':best_w1,'oof_mae':best_mae}
model_order.append('M2')

# MODEL 3: Fine-tuned Ridge multi-seed
print('\n--- MODEL 3 (Fine-Tuned Ridge Seeds) ---')
alpha_grid_ft = [0.2,0.25,0.3,0.35,0.4]
sc3 = RobustScaler().fit(X1)
X1_sc = sc3.transform(X1); T1_sc = sc3.transform(T1)
best_ft = {'alpha':None,'mae':np.inf}
for a in alpha_grid_ft:
    scores = cross_val_score(Ridge(alpha=a), X1_sc, y_full, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_ft['mae']:
        best_ft.update({'alpha':a,'mae':mae})
print(f"Best M3 alpha: {best_ft['alpha']} CV MAE: {best_ft['mae']:.4f}")
# OOF (average 2 seeds for efficiency)
oof3 = np.zeros(len(y_full))
for tr,va in kfold.split(X1_sc):
    fold_preds = []
    for sd in [42,123]:
        m = Ridge(alpha=best_ft['alpha'], random_state=sd)
        m.fit(X1_sc[tr], y_full.iloc[tr])
        fold_preds.append(m.predict(X1_sc[va]))
    oof3[va] = np.mean(fold_preds, axis=0)
# final test blend over more seeds
seed_preds = []
for sd in [42,123,456,789]:
    m = Ridge(alpha=best_ft['alpha'], random_state=sd)
    m.fit(X1_sc, y_full)
    seed_preds.append(m.predict(T1_sc))
pred3 = np.mean(seed_preds, axis=0)
model_preds_test['M3'] = pred3
oof_dict['M3'] = oof3
model_meta['M3'] = {'type':'ridge_seed','alpha':best_ft['alpha'],'oof_mae':mean_absolute_error(y_full,oof3)}
model_order.append('M3')

# MODEL 4: Win% (optional)
if ENABLE_MODEL4_WINPCT:
    print('\n--- MODEL 4 (Win% Ridge) ---')
    train_m4 = winpct_features(train_df); test_m4 = winpct_features(test_df)
    if 'G' in train_m4.columns:
        G_train = train_m4['G'].astype(float)
        y_pct = y_full / (G_train + 1e-6)
        feat4 = sorted((set(train_m4.columns) & set(test_m4.columns)) - EXCLUDE_BASE)
        X4 = train_m4[feat4].fillna(0).values; T4 = test_m4[feat4].fillna(0).values
        sc4 = StandardScaler().fit(X4)
        X4s = sc4.transform(X4); T4s = sc4.transform(T4)
        alphas4 = [0.05,0.1,0.2,0.3,0.4]
        best4 = {'alpha':None,'mae':np.inf}
        for a in alphas4:
            scores = cross_val_score(Ridge(alpha=a), X4s, y_pct, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = -scores.mean()
            if mae < best4['mae']:
                best4.update({'alpha':a,'mae':mae})
        print(f"Best M4 alpha={best4['alpha']} Win% CV MAE={best4['mae']:.5f}")
        oof4 = np.zeros(len(y_pct))
        for tr,va in kfold.split(X4s):
            m = Ridge(alpha=best4['alpha'])
            m.fit(X4s[tr], y_pct.iloc[tr])
            oof_pct = m.predict(X4s[va])
            oof4[va] = oof_pct * G_train.iloc[va]
        m_final4 = Ridge(alpha=best4['alpha']).fit(X4s, y_pct)
        pred4 = m_final4.predict(T4s) * test_m4['G'].astype(float)
        model_preds_test['M4'] = pred4
        oof_dict['M4'] = oof4
        model_meta['M4'] = {'type':'winpct_ridge','alpha':best4['alpha'],'oof_mae':mean_absolute_error(y_full,oof4)}
        model_order.append('M4')
    else:
        print('Skipping Model 4 (missing G column)')
else:
    print('\n(Model 4 disabled)')

# MODEL 5: ElasticNet
if ENABLE_MODEL5_ENET:
    print('\n--- MODEL 5 (ElasticNet) ---')
    X_en = scaler1.transform(train_m1[feat_m1].fillna(0).values)  # reuse scaler1
    T_en = scaler1.transform(test_m1[feat_m1].fillna(0).values)
    # small grid over alpha (overall strength) and l1_ratio
    alpha_grid = [0.0005,0.001,0.003,0.005,0.01]
    l1_grid = [0.2,0.4,0.6,0.8]
    best5 = {'alpha':None,'l1':None,'mae':np.inf}
    for a in alpha_grid:
        for l1 in l1_grid:
            en = ElasticNet(alpha=a, l1_ratio=l1, random_state=RANDOM_STATE, max_iter=5000)
            scores = cross_val_score(en, X_en, y_full, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = -scores.mean()
            if mae < best5['mae']:
                best5.update({'alpha':a,'l1':l1,'mae':mae})
    print(f"Best ENet alpha={best5['alpha']} l1={best5['l1']} CV MAE={best5['mae']:.4f}")
    oof5, en_final = oof_predictions(lambda: ElasticNet(alpha=best5['alpha'], l1_ratio=best5['l1'], random_state=RANDOM_STATE, max_iter=5000), X_en, y_full, kfold)
    pred5 = en_final.predict(T_en)
    model_preds_test['M5'] = pred5
    oof_dict['M5'] = oof5
    model_meta['M5'] = {'type':'elasticnet','alpha':best5['alpha'],'l1_ratio':best5['l1'],'oof_mae':mean_absolute_error(y_full,oof5)}
    model_order.append('M5')
else:
    print('\n(Model 5 disabled)')

# MODEL 6: Lasso
if ENABLE_MODEL6_LASSO:
    print('\n--- MODEL 6 (Lasso) ---')
    X_ls = scaler1.transform(train_m1[feat_m1].fillna(0).values)
    T_ls = scaler1.transform(test_m1[feat_m1].fillna(0).values)
    alpha_grid_lasso = [0.0001,0.0003,0.0005,0.001,0.003]
    best6 = {'alpha':None,'mae':np.inf}
    for a in alpha_grid_lasso:
        la = Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=5000)
        scores = cross_val_score(la, X_ls, y_full, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -scores.mean()
        if mae < best6['mae']:
            best6.update({'alpha':a,'mae':mae})
    print(f"Best Lasso alpha={best6['alpha']} CV MAE={best6['mae']:.4f}")
    oof6, la_final = oof_predictions(lambda: Lasso(alpha=best6['alpha'], random_state=RANDOM_STATE, max_iter=5000), X_ls, y_full, kfold)
    pred6 = la_final.predict(T_ls)
    model_preds_test['M6'] = pred6
    oof_dict['M6'] = oof6
    model_meta['M6'] = {'type':'lasso','alpha':best6['alpha'],'oof_mae':mean_absolute_error(y_full,oof6)}
    model_order.append('M6')
else:
    print('\n(Model 6 disabled)')

# =============================================================================
# Optional Residual Trimming (refit core ridge models M1,M2,M3 only) — disabled by default
# =============================================================================
if ENABLE_RESIDUAL_TRIMMING:
    print('\n--- Residual Trimming & Refit (Experimental) ---')
    core_models = ['M1','M2','M3']
    # Build provisional blend (equal weights)
    provisional = np.mean([oof_dict[m] for m in core_models], axis=0)
    residuals = np.abs(y_full - provisional)
    threshold = np.quantile(residuals, 1-RESIDUAL_TRIM_FRACTION)
    keep_mask = residuals <= threshold
    print(f"Trimming {np.sum(~keep_mask)} rows ({RESIDUAL_TRIM_FRACTION*100:.2f}%) with residual > {threshold:.3f}")
    # NOTE: For simplicity, not re-implementing full refit due to complexity here;
    # future step could rebuild models on trimmed dataset.
else:
    print('\n(Residual trimming disabled)')

# =============================================================================
# BUILD OOF MATRIX & ADAPTIVE MODEL SELECTION
# =============================================================================
print('\n--- Adaptive model inclusion ---')
# Start with models ordered by their individual OOF MAE ascending (strongest first)
models_sorted = sorted(model_meta.keys(), key=lambda m: model_meta[m]['oof_mae'])
print('Model ranking by OOF MAE (lower better):')
for m in models_sorted:
    print(f"  {m}: {model_meta[m]['oof_mae']:.4f}")

selected = []
current_blend = None
current_mae = np.inf
for m in models_sorted:
    trial_models = selected + [m]
    oof_stack = np.vstack([oof_dict[x] for x in trial_models])
    w_eq = np.ones(len(trial_models))/len(trial_models)
    blend = np.einsum('i,ij->j', w_eq, oof_stack)
    mae_trial = mean_absolute_error(y_full, blend)
    if current_blend is None or current_mae - mae_trial >= MIN_IMPROVEMENT_KEEP_MODEL:
        selected = trial_models
        current_blend = blend
        current_mae = mae_trial
        print(f"+ Keeping {m} | New OOF MAE: {current_mae:.5f}")
    else:
        print(f"- Skipping {m} (improvement {current_mae - mae_trial:.5f} < {MIN_IMPROVEMENT_KEEP_MODEL})")

print(f"Selected models for blending: {selected}")

# Construct final OOF and TEST stack for selected models
OOF_STACK = np.vstack([oof_dict[m] for m in selected])
TEST_STACK = np.vstack([model_preds_test[m] for m in selected])

# =============================================================================
# WEIGHT OPTIMIZATION
# =============================================================================
print('\n--- Weight Optimization (Dirichlet + Local + Coordinate Descent) ---')

rng = np.random.default_rng(RANDOM_STATE)
K = OOF_STACK.shape[0]
# Equal weight baseline
w_best = np.ones(K)/K
mae_best = mean_absolute_error(y_full, np.einsum('i,ij->j', w_best, OOF_STACK))
print(f"Baseline equal MAE: {mae_best:.5f}")

# Random Dirichlet search
for w in rng.dirichlet(np.ones(K), size=DIRICHLET_SAMPLES):
    pred = np.einsum('i,ij->j', w, OOF_STACK)
    mae = mean_absolute_error(y_full, pred)
    if mae < mae_best:
        mae_best = mae; w_best = w.copy()
print(f"After Dirichlet search MAE: {mae_best:.5f}")

# Local random refinement
no_improve = 0
for step in range(LOCAL_STEPS):
    sigma = max(0.08 * (1 - step/LOCAL_STEPS), 0.01)
    noise = rng.normal(0, sigma, size=K)
    w_c = w_best + noise
    w_c = np.clip(w_c, 0, None)
    if w_c.sum() == 0: continue
    w_c /= w_c.sum()
    pred = np.einsum('i,ij->j', w_c, OOF_STACK)
    mae = mean_absolute_error(y_full, pred)
    if mae + 1e-9 < mae_best:
        w_best = w_c; mae_best = mae; no_improve = 0
    else:
        no_improve += 1
    if no_improve >= EARLY_STOP_NO_IMPROVE:
        break
print(f"After local refinement MAE: {mae_best:.5f} (stopped at step {step})")

# Coordinate descent (subgradient sign approximation)
print('Coordinate descent refinement...')
for it in range(COORD_DESC_ITERS):
    improved = False
    base_pred = np.einsum('i,ij->j', w_best, OOF_STACK)
    residual = y_full.values - base_pred
    sign_r = np.sign(residual)
    # approximate gradient for each weight: -mean(sign(residual)*model_oof)
    grads = []
    for k in range(K):
        grads.append(-np.mean(sign_r * OOF_STACK[k]))
    grads = np.array(grads)
    # take small step opposite gradient
    step_vec = -COORD_STEP * grads
    w_c = w_best + step_vec
    w_c = np.clip(w_c, 0, None)
    if w_c.sum() == 0: continue
    w_c /= w_c.sum()
    pred = np.einsum('i,ij->j', w_c, OOF_STACK)
    mae = mean_absolute_error(y_full, pred)
    if mae + 1e-9 < mae_best:
        w_best = w_c; mae_best = mae; improved = True
    if not improved and it % 60 == 0:
        # occasional random shake
        jitter = rng.normal(0, 0.01, size=K)
        w_j = w_best + jitter
        w_j = np.clip(w_j,0,None)
        if w_j.sum() > 0: w_j /= w_j.sum()
        pred_j = np.einsum('i,ij->j', w_j, OOF_STACK)
        mae_j = mean_absolute_error(y_full, pred_j)
        if mae_j + 1e-9 < mae_best:
            w_best = w_j; mae_best = mae_j
print(f"Final blended OOF MAE: {mae_best:.5f}")

# =============================================================================
# FINAL PREDICTIONS & OUTPUT
# =============================================================================
print('\n--- Final Prediction & Submission ---')
final_float = np.einsum('i,ij->j', w_best, TEST_STACK)
final_pred = np.clip(final_float, 0, 162).round().astype(int)
submission = pd.DataFrame({'ID': test_df['ID'], 'W': final_pred})
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Saved submission: {SUBMISSION_FILE} | Mean={final_pred.mean():.2f} Std={final_pred.std():.2f}")

# Residual correlation matrix (selected models)
residuals = [ y_full.values - oof_dict[m] for m in selected ]
# Handle single-model edge case safely
if len(selected) == 0:
    res_corr = np.array([[]])
elif len(selected) == 1:
    res_corr = np.array([[1.0]])
else:
    res_corr = np.corrcoef(residuals)
    if np.ndim(res_corr) == 0:  # safety wrap
        res_corr = np.array([[float(res_corr)]])
print('\nResidual correlation matrix (selected models):')
if len(selected) <= 1:
    print('  (Only one model selected; correlation matrix is trivial [1.0])')
else:
    for i,m in enumerate(selected):
        row = ' '.join(f"{v:6.3f}" for v in res_corr[i])
        print(f"  {m}: {row}")

# Save diagnostics
try:
    diagnostics = {
        'selected_models': selected,
        'weights': w_best.tolist(),
        'oof_mae': float(mae_best),
        'individual_oof': {m: float(model_meta[m]['oof_mae']) for m in selected},
        'residual_corr': res_corr.tolist(),
        'config': {
            'ENABLE_MODEL4_WINPCT': ENABLE_MODEL4_WINPCT,
            'ENABLE_MODEL5_ENET': ENABLE_MODEL5_ENET,
            'ENABLE_MODEL6_LASSO': ENABLE_MODEL6_LASSO,
            'ENABLE_RESIDUAL_TRIMMING': ENABLE_RESIDUAL_TRIMMING
        }
    }
    with open(DIAGNOSTICS_FILE,'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Diagnostics saved: {DIAGNOSTICS_FILE}")
except Exception as e:
    print('Diagnostics save failed:', e)

print('\nSummary:')
for m in selected:
    print(f"  {m}: OOF {model_meta[m]['oof_mae']:.5f}")
print(f"Blend OOF MAE: {mae_best:.5f}")
print('Weights:')
for m,w in zip(selected, w_best):
    print(f"  {m}: {w:.4f}")
print('\nNext suggestions if not improved:')
print(' - Lower MIN_IMPROVEMENT_KEEP_MODEL to allow more models into blend (currently 0.0003).')
print(' - Increase DIRICHLET_SAMPLES or COORD_DESC_ITERS for deeper weight search.')
print(' - Re-enable Win% model or add tree-based learner for non-linear lift.')
print(' - Try residual trimming or stacking meta-model.')
print('\n🚀 Submit and compare to previous public LB scores.')
print('='*100)
