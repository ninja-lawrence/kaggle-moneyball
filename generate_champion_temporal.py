"""
═══════════════════════════════════════════════════════════════════════════════
TEMPORAL / SHIFT-AWARE RECOVERY SCRIPT
═══════════════════════════════════════════════════════════════════════════════
Issue Observed:
  Strong random KFold OOF MAE (~2.77) but public LB ≈ 3.02 ⇒ Classic sign of
  temporal / distribution shift: random shuffling leaks information from future
  eras into training folds, producing optimistic OOF scores.

Goals:
  1. Use strictly time-aware validation (chronological splits + decade GroupKFold).
  2. Run adversarial validation (train vs test classifier) to quantify shift.
  3. Compute importance weights via p(test|x)/(1 - p(test|x)) and reweight ridge.
  4. Compare MAE under:
       • Random KFold (baseline optimistic)
       • Chronological splits (Year-based folds)
       • GroupKFold by decade
       • Importance-weighted ridge (chronological)
  5. Produce a conservative submission using the time-aware weighted model.
  6. Optional decade residual calibration (shrunk median residual correction).

Output:
  submission_champion_temporal.csv
  temporal_diagnostics.json (CV metrics, shift scores, weight stats)

Next Extensions (not implemented yet):
  - Add tree model with time-aware CV only.
  - Per-era feature interactions & partial pooling.
  - Quantile calibration of residual distribution.

Run:
  python generate_champion_temporal.py
═══════════════════════════════════════════════════════════════════════════════
Date: 2025-10-07
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json, warnings
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST  = 'data/test.csv'
SUBMISSION_FILE = 'submission_champion_temporal.csv'
DIAGNOSTICS_FILE = 'temporal_diagnostics.json'

N_FOLDS_RANDOM = 10
N_FOLDS_GROUP  = 10
N_CHRONO_SPLITS = 8          # number of chronological folds
RANDOM_STATE = 42
ALPHA_GRID = [0.2,0.25,0.3,0.35,0.4,0.5,0.75,1.0,1.5,2.0]
ADVERSARIAL_MODEL = 'hgb'    # 'hgb' or 'logreg'
IMPORTANCE_WEIGHT_CLIP = 5.0
DECADE_CALIBRATION = True
DECADE_CALIBRATION_SHRINK = 0.5
CALIBRATION_MAX_ABS = 1.2

np.random.seed(RANDOM_STATE)

# =============================================================================
# DATA
# =============================================================================
train_df = pd.read_csv(DATA_TRAIN)
test_df  = pd.read_csv(DATA_TEST)
y = train_df['W'].astype(float)

if 'yearID' not in train_df.columns:
    raise ValueError('yearID column required for temporal validation.')

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
    # Clean
    df = df.replace([np.inf,-np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum():
            df[col]=df[col].fillna(df[col].median())
    return df

EXCLUDE = {
    'W','ID','teamID','yearID','year_label','decade_label','win_bins',
    'decade_1910','decade_1920','decade_1930','decade_1940','decade_1950','decade_1960',
    'decade_1970','decade_1980','decade_1990','decade_2000','decade_2010',
    'era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','mlb_rpg'
}

train_feat = build_features(train_df)
test_feat  = build_features(test_df)

features = sorted((set(train_feat.columns) & set(test_feat.columns)) - EXCLUDE)
X = train_feat[features].values
X_test = test_feat[features].values

# Scaler (robust to outliers)
scaler = RobustScaler().fit(X)
X_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)

# Helper: choose alpha via time-aware CV (GroupKFold by decade)
train_years = train_df['yearID'].values
train_decade = (train_years//10)*10

def evaluate_alpha_time(alpha: float):
    # GroupKFold by decade
    gkf = GroupKFold(n_splits=min(N_FOLDS_GROUP, len(np.unique(train_decade))))
    maes = []
    for tr, va in gkf.split(X_scaled, y, groups=train_decade):
        m = Ridge(alpha=alpha)
        m.fit(X_scaled[tr], y.iloc[tr])
        p = m.predict(X_scaled[va])
        maes.append(mean_absolute_error(y.iloc[va], p))
    return np.mean(maes)

alpha_scores = []
for a in ALPHA_GRID:
    alpha_scores.append((a, evaluate_alpha_time(a)))
alpha_scores.sort(key=lambda x: x[1])
best_alpha, best_time_mae = alpha_scores[0]

# Random KFold (reference optimistic)
rand_kf = KFold(n_splits=N_FOLDS_RANDOM, shuffle=True, random_state=RANDOM_STATE)
rand_maes = []
for tr,va in rand_kf.split(X_scaled):
    m = Ridge(alpha=best_alpha)
    m.fit(X_scaled[tr], y.iloc[tr])
    rand_maes.append(mean_absolute_error(y.iloc[va], m.predict(X_scaled[va])))
rand_mae_mean = np.mean(rand_maes)

# Chronological splits (progressive expanding window)
chron_maes = []
sorted_idx = np.argsort(train_years)
X_chron = X_scaled[sorted_idx]
y_chron = y.iloc[sorted_idx].reset_index(drop=True)
years_sorted = train_years[sorted_idx]
unique_years = np.unique(years_sorted)
# Create roughly N_CHRONO_SPLITS folds by slicing years
year_splits = np.array_split(unique_years, N_CHRONO_SPLITS)
# We will use each segment as validation with all previous as train (skip first)
for i in range(1, len(year_splits)):
    val_years = year_splits[i]
    train_years_set = np.concatenate(year_splits[:i])
    tr_mask = np.isin(years_sorted, train_years_set)
    va_mask = np.isin(years_sorted, val_years)
    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        continue
    m = Ridge(alpha=best_alpha)
    m.fit(X_chron[tr_mask], y_chron[tr_mask])
    chron_maes.append(mean_absolute_error(y_chron[va_mask], m.predict(X_chron[va_mask])))
chron_mae_mean = float(np.mean(chron_maes)) if chron_maes else None

# GroupKFold OOF preds for best alpha (for calibration & residual analysis)
gkf = GroupKFold(n_splits=min(N_FOLDS_GROUP, len(np.unique(train_decade))))
oof_time = np.zeros(len(y))
for tr,va in gkf.split(X_scaled, y, groups=train_decade):
    m = Ridge(alpha=best_alpha)
    m.fit(X_scaled[tr], y.iloc[tr])
    oof_time[va] = m.predict(X_scaled[va])
base_time_oof_mae = mean_absolute_error(y, oof_time)

# =============================================================================
# ADVERSARIAL VALIDATION
# =============================================================================
print('Running adversarial validation...')
train_adv = X_scaled
test_adv = X_test_scaled
adv_X = np.vstack([train_adv, test_adv])
adv_y = np.concatenate([np.zeros(len(train_adv)), np.ones(len(test_adv))])

if ADVERSARIAL_MODEL == 'hgb':
    clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=250,
                                         min_samples_leaf=30, l2_regularization=0.6,
                                         random_state=RANDOM_STATE)
else:
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

clf.fit(adv_X, adv_y)
prob_train = clf.predict_proba(train_adv)[:,1]  # p(test|x) for training rows
prob_test  = clf.predict_proba(test_adv)[:,1]
auc_adv = roc_auc_score(adv_y, clf.predict_proba(adv_X)[:,1])
print(f"Adversarial AUC (train vs test separability): {auc_adv:.4f}")

# Importance weights (clip extremes)
w_importance = prob_train / (1 - prob_train + 1e-6)
w_importance = np.clip(w_importance, 0, IMPORTANCE_WEIGHT_CLIP)

# Weighted GroupKFold evaluation
w_maes = []
for tr,va in gkf.split(X_scaled, y, groups=train_decade):
    m = Ridge(alpha=best_alpha)
    m.fit(X_scaled[tr], y.iloc[tr], sample_weight=w_importance[tr])
    pred = m.predict(X_scaled[va])
    w_maes.append(mean_absolute_error(y.iloc[va], pred))
weighted_time_mae = np.mean(w_maes)

# Train final importance-weighted model on all training data
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_scaled, y, sample_weight=w_importance)
final_pred_float = final_model.predict(X_test_scaled)

# =============================================================================
# DECADE CALIBRATION (optional)
# =============================================================================
if DECADE_CALIBRATION and 'yearID' in train_df.columns and 'yearID' in test_df.columns:
    decades = train_decade
    residuals = y - oof_time
    decade_offsets = {}
    for dec in sorted(np.unique(decades)):
        mask = decades == dec
        if mask.sum() < 8:
            continue
        med = residuals[mask].median()
        adj = med * DECADE_CALIBRATION_SHRINK
        adj = float(np.clip(adj, -CALIBRATION_MAX_ABS, CALIBRATION_MAX_ABS))
        decade_offsets[int(dec)] = adj
    test_decades = (test_df['yearID'].values//10)*10
    offsets_vec = np.array([decade_offsets.get(int(d), 0.0) for d in test_decades])
    final_pred_float_cal = final_pred_float + offsets_vec
    decade_calibration_note = 'applied'
else:
    # Fallback: no calibration (likely missing yearID in test set) – warn once
    decade_offsets = {}
    final_pred_float_cal = final_pred_float
    decade_calibration_note = 'skipped_missing_yearID' if 'yearID' not in test_df.columns else 'disabled'

# Final rounding
final_pred = np.clip(final_pred_float_cal, 0, 162).round().astype(int)

pd.DataFrame({'ID': test_df['ID'], 'W': final_pred}).to_csv(SUBMISSION_FILE, index=False)

# =============================================================================
# DIAGNOSTICS
# =============================================================================
try:
    diagnostics = {
        'best_alpha': best_alpha,
        'time_mae_groupkfold': float(best_time_mae),
        'random_mae_mean': float(rand_mae_mean),
        'chron_mae_mean': chron_mae_mean,
        'group_oof_mae_full': float(base_time_oof_mae),
        'importance_weighted_time_mae': float(weighted_time_mae),
        'adversarial_auc': float(auc_adv),
        'importance_weight_stats': {
            'mean': float(w_importance.mean()),
            'std': float(w_importance.std()),
            'min': float(w_importance.min()),
            'max': float(w_importance.max())
        },
        'decade_offsets': decade_offsets,
        'decade_calibration_status': decade_calibration_note,
        'feature_count': len(features)
    }
    with open(DIAGNOSTICS_FILE,'w') as f:
        json.dump(diagnostics, f, indent=2)
except Exception as e:
    print('Diagnostics save failed:', e)

# =============================================================================
# SUMMARY PRINT
# =============================================================================
print('================ TEMPORAL MODEL SUMMARY ================')
print(f"Optimistic Random KFold MAE: {rand_mae_mean:.4f}")
print(f"Group (Decade) MAE (alpha search): {best_time_mae:.4f}")
print(f"Chronological expanding MAE: {chron_mae_mean:.4f}")
print(f"Group OOF (refit) MAE: {base_time_oof_mae:.4f}")
print(f"Importance-weighted Group MAE: {weighted_time_mae:.4f}")
print(f"Adversarial AUC: {auc_adv:.4f}")
print(f"Importance weight stats: mean={w_importance.mean():.3f} max={w_importance.max():.3f}")
print(f"Decade calibration: {decade_calibration_note}")
print(f"Submission saved: {SUBMISSION_FILE}")
print('=========================================================')
