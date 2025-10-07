"""
═══════════════════════════════════════════════════════════════════════════════
TEMPORAL DRIFT MITIGATION SCRIPT
═══════════════════════════════════════════════════════════════════════════════
Context:
  Prior temporal run showed:
    • Random KFold MAE ≈ 2.7750 (over-optimistic)
    • Group (decade) MAE ≈ 2.83
    • Chronological MAE ≈ 3.00 (closest to LB ≈ 3.02)
    • Adversarial AUC ≈ 0.9999 ⇒ Severe distribution shift.
    • Importance-weighted ridge worsened MAE (2.8555).

Goal:
  Reduce chronological MAE by removing most drifted (unstable) features and
  increasing regularization, producing a conservative submission.

Strategy Implemented:
  1. Build base feature set (same transformations as before).
  2. Train adversarial classifier (train vs test) to get feature importances.
  3. Evaluate multiple retention ratios (retain least-drift features):
       RETENTION_RATIOS = [0.4, 0.5, 0.6, 0.7, 0.8]
     For each, select stable features (lowest importance) and re-run:
       • Chronological expanding window MAE
       • Decade GroupKFold MAE
     Track best ratio by chronological MAE (ties broken by group MAE).
  4. Expanded ALPHA_GRID including stronger shrinkage (up to 12) for stability.
  5. Final model: Ridge on best ratio + best alpha (chronological criterion).
  6. Optional blend: convex combination with full-feature temporal model if it
     improves chronological MAE (simple 11-point grid over lambda).
  7. Output submission + diagnostics JSON with per-ratio metrics and chosen setup.

Outputs:
  submission_champion_temporal_drift.csv
  temporal_drift_diagnostics.json

Next extensions (not yet implemented):
  - Residual tree on stable feature residuals (time-aware CV)
  - Feature cluster smoothing (PCA within stable subset) to reduce noise
  - Quantile calibration to reduce tails

Run:
  python generate_champion_temporal_drift.py
═══════════════════════════════════════════════════════════════════════════════
Date: 2025-10-07
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json, warnings
from typing import List, Dict, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST  = 'data/test.csv'
SUBMISSION_FILE = 'submission_champion_temporal_drift.csv'
DIAGNOSTICS_FILE = 'temporal_drift_diagnostics.json'

RANDOM_STATE = 42
RETENTION_RATIOS = [0.4, 0.5, 0.6, 0.7, 0.8]
ALPHA_GRID = [0.2,0.3,0.4,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,10.0,12.0]
N_CHRONO_SPLITS = 8
IMPORTANCE_TOP_CLIP = 0.02  # ignore extreme small noise at tail for stability
BLEND_WITH_FULL_MODEL = True
BLEND_LAMBDAS = np.linspace(0,1,11)  # lambda=0 => stable-only, 1 => full model

np.random.seed(RANDOM_STATE)

# =============================================================================
# LOAD DATA
# =============================================================================
train_df = pd.read_csv(DATA_TRAIN)
test_df  = pd.read_csv(DATA_TEST)
y = train_df['W'].astype(float)
if 'yearID' not in train_df.columns:
    raise ValueError('yearID required for temporal drift script.')

# =============================================================================
# FEATURE ENGINEERING (same base transformations)
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
            df[col]=df[col].fillna(df[col].median())
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

scaler_full = RobustScaler().fit(X_full)
X_full_s = scaler_full.transform(X_full)
X_test_full_s = scaler_full.transform(X_test_full)

# =============================================================================
# ADVERSARIAL DRIFT MODEL
# =============================================================================
print('Training adversarial classifier for drift feature importances...')
adv_X = np.vstack([X_full_s, X_test_full_s])
adv_y = np.concatenate([np.zeros(len(X_full_s)), np.ones(len(X_test_full_s))])
clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=350,
                                     l2_regularization=0.6, min_samples_leaf=30,
                                     random_state=RANDOM_STATE)
clf.fit(adv_X, adv_y)
adv_probs = clf.predict_proba(adv_X)[:,1]
auc_adv = roc_auc_score(adv_y, adv_probs)
print(f"Adversarial AUC: {auc_adv:.4f}")

# --- Permutation-based feature importance (since HistGradientBoostingClassifier lacks feature_importances_) ---
print('Computing permutation importances (drift)...')
from sklearn.utils import shuffle as sk_shuffle
baseline_auc = roc_auc_score(adv_y, adv_probs)
perm_importances = []
# Work on a subsample for speed if very large
max_perm_rows = 20000
if adv_X.shape[0] > max_perm_rows:
    idx_sample = np.random.choice(np.arange(adv_X.shape[0]), size=max_perm_rows, replace=False)
    X_perm_base = adv_X[idx_sample].copy()
    y_perm_base = adv_y[idx_sample].copy()
    base_probs_sample = clf.predict_proba(X_perm_base)[:,1]
    baseline_auc = roc_auc_score(y_perm_base, base_probs_sample)
else:
    X_perm_base = adv_X.copy()
    y_perm_base = adv_y.copy()
    base_probs_sample = adv_probs

rng = np.random.default_rng(RANDOM_STATE)
for j, feat in enumerate(all_features):
    X_shuffled = X_perm_base.copy()
    # Shuffle column j only within sample
    col = X_shuffled[:, j].copy()
    rng.shuffle(col)
    X_shuffled[:, j] = col
    probs_shuf = clf.predict_proba(X_shuffled)[:,1]
    auc_shuf = roc_auc_score(y_perm_base, probs_shuf)
    importance = baseline_auc - auc_shuf  # drop in AUC
    perm_importances.append(importance)

importances = np.array(perm_importances)
# If all importances very small (rare), fallback to absolute Pearson corr with probs
if np.allclose(importances, 0):
    print('Permutation importances near zero; using correlation proxy.')
    # compute correlation only on training portion to avoid test leakage effect
    train_probs = clf.predict_proba(X_full_s)[:,1]
    corrs = []
    for j in range(X_full_s.shape[1]):
        xj = X_full_s[:, j]
        if np.std(xj) < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(abs(np.corrcoef(xj, train_probs)[0,1]))
    importances = np.array(corrs)

feat_importance_pairs = list(zip(all_features, importances))
# Sort descending by drift (higher importance = more drift)
feat_importance_pairs.sort(key=lambda x: x[1], reverse=True)

# Build mapping for easy retention selection
feat_to_importance = dict(feat_importance_pairs)

# =============================================================================
# EVALUATION HELPERS
# =============================================================================
train_years = train_df['yearID'].values
sorted_idx = np.argsort(train_years)
unique_years = np.unique(train_years)
chrono_year_splits = np.array_split(unique_years, N_CHRONO_SPLITS)

def chronological_mae(Xm: np.ndarray) -> float:
    X_sorted = Xm[sorted_idx]
    y_sorted = y.iloc[sorted_idx].reset_index(drop=True)
    years_sorted = train_years[sorted_idx]
    maes = []
    for i in range(1, len(chrono_year_splits)):
        val_years = chrono_year_splits[i]
        train_years_concat = np.concatenate(chrono_year_splits[:i])
        tr_mask = np.isin(years_sorted, train_years_concat)
        va_mask = np.isin(years_sorted, val_years)
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue
        best_alpha_local = None
        best_local_mae = np.inf
        for a in ALPHA_GRID:
            m = Ridge(alpha=a)
            m.fit(X_sorted[tr_mask], y_sorted[tr_mask])
            pred = m.predict(X_sorted[va_mask])
            mae = mean_absolute_error(y_sorted[va_mask], pred)
            if mae < best_local_mae:
                best_local_mae = mae
                best_alpha_local = a
        maes.append(best_local_mae)
    return float(np.mean(maes)) if maes else np.inf

def group_mae(Xm: np.ndarray) -> Tuple[float, float]:
    decades = (train_years//10)*10
    gkf = GroupKFold(n_splits=min(10, len(np.unique(decades))))
    maes = []
    alpha_best = None
    alpha_best_mae = np.inf
    # Preselect alpha via quick loop using same folds
    for a in ALPHA_GRID:
        fold_maes = []
        for tr,va in gkf.split(Xm, y, groups=decades):
            m = Ridge(alpha=a)
            m.fit(Xm[tr], y.iloc[tr])
            fold_maes.append(mean_absolute_error(y.iloc[va], m.predict(Xm[va])))
        avg = np.mean(fold_maes)
        if avg < alpha_best_mae:
            alpha_best_mae = avg
            alpha_best = a
    # recompute OOF with best alpha for reporting
    oof = np.zeros(len(y))
    for tr,va in gkf.split(Xm, y, groups=decades):
        m = Ridge(alpha=alpha_best)
        m.fit(Xm[tr], y.iloc[tr])
        oof[va] = m.predict(Xm[va])
    return alpha_best_mae, alpha_best

# =============================================================================
# RETENTION RATIO SEARCH
# =============================================================================
ratio_results = []
print('Evaluating retention ratios for stable features...')
for ratio in RETENTION_RATIOS:
    keep_count = max(3, int(len(all_features)*ratio))
    # Keep lowest-drift features ⇒ sort ascending by importance
    ascending = sorted(feat_importance_pairs, key=lambda x: x[1])
    kept_feats = [f for f,_ in ascending[:keep_count]]
    X_sub = train_feat_full[kept_feats].values
    T_sub = test_feat_full[kept_feats].values
    scaler_sub = RobustScaler().fit(X_sub)
    X_sub_s = scaler_sub.transform(X_sub)
    # Chronological MAE
    chrono_score = chronological_mae(X_sub_s)
    group_score, alpha_group = group_mae(X_sub_s)
    ratio_results.append({
        'ratio': ratio,
        'kept_features': kept_feats,
        'kept_count': len(kept_feats),
        'chronological_mae': chrono_score,
        'group_mae': group_score,
        'group_best_alpha': alpha_group
    })
    print(f"Ratio {ratio:.2f} | kept={len(kept_feats):3d} | chrono={chrono_score:.4f} | group={group_score:.4f} | alpha={alpha_group}")

# Select best by chronological MAE then by group MAE
def ratio_key(d):
    return (d['chronological_mae'], d['group_mae'])
ratio_results.sort(key=ratio_key)
best_ratio_entry = ratio_results[0]
print('\nBest ratio selected:', best_ratio_entry['ratio'], 'chrono_mae=', best_ratio_entry['chronological_mae'], 'group_mae=', best_ratio_entry['group_mae'])

# =============================================================================
# FINAL MODEL (STABLE FEATURES)
# =============================================================================
kept_feats = best_ratio_entry['kept_features']
X_stable = train_feat_full[kept_feats].values
T_stable = test_feat_full[kept_feats].values
scaler_stable = RobustScaler().fit(X_stable)
X_stable_s = scaler_stable.transform(X_stable)
T_stable_s = scaler_stable.transform(T_stable)

# Determine best alpha via chronological folds again restricted to kept features
alpha_chrono_scores = []
sorted_year_idx = np.argsort(train_years)
X_sorted_stable = X_stable_s[sorted_year_idx]
y_sorted = y.iloc[sorted_year_idx].reset_index(drop=True)
years_sorted = train_years[sorted_year_idx]
for a in ALPHA_GRID:
    maes = []
    for i in range(1, len(chrono_year_splits)):
        val_years = chrono_year_splits[i]
        train_years_concat = np.concatenate(chrono_year_splits[:i])
        tr_mask = np.isin(years_sorted, train_years_concat)
        va_mask = np.isin(years_sorted, val_years)
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue
        m = Ridge(alpha=a)
        m.fit(X_sorted_stable[tr_mask], y_sorted[tr_mask])
        pred = m.predict(X_sorted_stable[va_mask])
        maes.append(mean_absolute_error(y_sorted[va_mask], pred))
    alpha_chrono_scores.append((a, np.mean(maes) if maes else np.inf))
alpha_chrono_scores.sort(key=lambda x: x[1])
best_alpha_stable, best_alpha_stable_chrono = alpha_chrono_scores[0]
print(f"Stable feature best alpha (chronological): {best_alpha_stable} chrono_MAE={best_alpha_stable_chrono:.4f}")

# Fit final stable model
stable_model = Ridge(alpha=best_alpha_stable)
stable_model.fit(X_stable_s, y)
stable_pred_float = stable_model.predict(T_stable_s)

# Also train full-feature temporal model for optional blend (same alpha as best group earlier or re-estimate)
full_alpha_scores = []
for a in ALPHA_GRID:
    maes = []
    for i in range(1, len(chrono_year_splits)):
        val_years = chrono_year_splits[i]
        train_years_concat = np.concatenate(chrono_year_splits[:i])
        tr_mask = np.isin(years_sorted, train_years_concat)
        va_mask = np.isin(years_sorted, val_years)
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue
        m = Ridge(alpha=a)
        m.fit(X_full_s[sorted_year_idx][tr_mask], y_sorted[tr_mask])
        pred = m.predict(X_full_s[sorted_year_idx][va_mask])
        maes.append(mean_absolute_error(y_sorted[va_mask], pred))
    full_alpha_scores.append((a, np.mean(maes) if maes else np.inf))
full_alpha_scores.sort(key=lambda x: x[1])
best_alpha_full, best_alpha_full_chrono = full_alpha_scores[0]
print(f"Full feature best alpha (chronological): {best_alpha_full} chrono_MAE={best_alpha_full_chrono:.4f}")
full_model = Ridge(alpha=best_alpha_full).fit(X_full_s, y)
full_pred_float = full_model.predict(X_test_full_s)

# Optional blend stable vs full based on chronological OOB style predictions
if BLEND_WITH_FULL_MODEL:
    # Build pseudo-OOB chronological predictions for each model
    def chrono_oof(Xm_s: np.ndarray, alpha: float) -> np.ndarray:
        preds = np.zeros(len(y))
        for i in range(1, len(chrono_year_splits)):
            val_years = chrono_year_splits[i]
            train_years_concat = np.concatenate(chrono_year_splits[:i])
            tr_mask = np.isin(train_years, train_years_concat)
            va_mask = np.isin(train_years, val_years)
            if tr_mask.sum()==0 or va_mask.sum()==0:
                continue
            m = Ridge(alpha=alpha)
            m.fit(Xm_s[tr_mask], y.iloc[tr_mask])
            preds[va_mask] = m.predict(Xm_s[va_mask])
        return preds
    oof_stable = chrono_oof(X_stable_s, best_alpha_stable)
    oof_full   = chrono_oof(X_full_s,   best_alpha_full)
    best_blend = {'lambda':0.0,'mae':np.inf}
    for lam in BLEND_LAMBDAS:
        blend = lam*oof_full + (1-lam)*oof_stable
        mae = mean_absolute_error(y, blend)
        if mae < best_blend['mae']:
            best_blend.update({'lambda':lam,'mae':mae})
    print(f"Blend tuning stable/full: best_lambda={best_blend['lambda']:.2f} chrono_like_MAE={best_blend['mae']:.4f}")
    final_float = best_blend['lambda']*full_pred_float + (1-best_blend['lambda'])*stable_pred_float
    blend_lambda = best_blend['lambda']
else:
    final_float = stable_pred_float
    blend_lambda = None

final_pred = np.clip(final_float, 0, 162).round().astype(int)
pd.DataFrame({'ID': test_df['ID'], 'W': final_pred}).to_csv(SUBMISSION_FILE, index=False)
print(f"Saved submission: {SUBMISSION_FILE}")

# =============================================================================
# DIAGNOSTICS SAVE
# =============================================================================
try:
    diag = {
        'adversarial_auc': float(auc_adv),
        'feature_importances': feat_importance_pairs[:50],  # top 50 drift features
        'ratio_results': ratio_results,
        'best_ratio': best_ratio_entry,
        'best_alpha_stable': best_alpha_stable,
        'best_alpha_stable_chrono_mae': best_alpha_stable_chrono,
        'best_alpha_full': best_alpha_full,
        'best_alpha_full_chrono_mae': best_alpha_full_chrono,
        'blend_lambda': blend_lambda,
        'submission_file': SUBMISSION_FILE,
        'stable_feature_count': len(kept_feats),
        'total_feature_count': len(all_features)
    }
    with open(DIAGNOSTICS_FILE,'w') as f:
        json.dump(diag, f, indent=2)
    print(f"Diagnostics saved: {DIAGNOSTICS_FILE}")
except Exception as e:
    print('Diagnostics save failed:', e)

# =============================================================================
# SUMMARY PRINT
# =============================================================================
print('=============== DRIFT MITIGATION SUMMARY ===============')
print(f"Adversarial AUC: {auc_adv:.4f}")
print(f"Best retention ratio: {best_ratio_entry['ratio']:.2f} | kept {best_ratio_entry['kept_count']} features")
print(f"Stable alpha: {best_alpha_stable} | stable chrono MAE: {best_alpha_stable_chrono:.4f}")
print(f"Full alpha: {best_alpha_full} | full chrono MAE: {best_alpha_full_chrono:.4f}")
if blend_lambda is not None:
    print(f"Blend lambda (full portion): {blend_lambda:.2f}")
print(f"Submission: {SUBMISSION_FILE}")
print('========================================================')
