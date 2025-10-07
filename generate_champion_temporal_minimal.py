"""
═══════════════════════════════════════════════════════════════════════════════
TEMPORAL MINIMAL BASELINE (Recovery)
═══════════════════════════════════════════════════════════════════════════════
Purpose:
  Leaderboard score regressed sharply (≈8.07). This script provides a *very*
  conservative temporal model to re-establish a stable baseline before layering
  complexity.

Characteristics:
  • Uses ONLY stable engineered features (no adversarial weighting, no stacking).
  • Chronological expanding-window folds for alpha selection (prevents leakage).
  • Multi-seed ridge averaging for variance reduction.
  • Final predictions: single rounding & clipping only.
  • Diagnostics JSON with distribution summaries and chronological MAE.

Run:
  python generate_champion_temporal_minimal.py
Output:
  submission_champion_temporal_minimal.csv
  temporal_minimal_diagnostics.json

Next (after verifying improvement):
  1. Add second model (full features) and 2-model chronological blend.
  2. Introduce elastic net if stable MAE plateau.
  3. Only then consider tree residual or drift pruning.
═══════════════════════════════════════════════════════════════════════════════
Date: 2025-10-07
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json, warnings

# Attempt to import sklearn; if unavailable, define lightweight fallbacks so the
# script can run in constrained environments (e.g., Kaggle image missing lib or
# local environment not yet provisioned). The fallback Ridge is a closed-form
# solver for L2-regularized least squares; RobustScaler uses median/IQR.
try:
    from sklearn.linear_model import Ridge  # type: ignore
    from sklearn.preprocessing import RobustScaler  # type: ignore
    from sklearn.metrics import mean_absolute_error  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    SKLEARN_AVAILABLE = False

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    class RobustScaler:  # minimal drop-in subset
        def __init__(self):
            self.center_ = None
            self.scale_ = None

        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.center_ = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            # Avoid divide by zero
            iqr[iqr == 0] = 1.0
            self.scale_ = iqr
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.center_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class Ridge:  # minimal deterministic ridge (no randomness)
        def __init__(self, alpha=1.0, random_state=None):
            self.alpha = alpha
            self.coef_ = None
            self.intercept_ = None

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            # Add bias column
            Xb = np.c_[np.ones(X.shape[0]), X]
            # Closed form: (X^T X + alpha*I)^{-1} X^T y
            XtX = Xb.T @ Xb
            # Do not regularize intercept (set top-left to 0)
            reg = self.alpha * np.eye(XtX.shape[0])
            reg[0,0] = 0.0
            w = np.linalg.solve(XtX + reg, Xb.T @ y)
            self.intercept_ = w[0]
            self.coef_ = w[1:]
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return self.intercept_ + X @ self.coef_

    print("[INFO] sklearn not found. Using lightweight numpy fallbacks for Ridge & RobustScaler.")

warnings.filterwarnings('ignore')

# ============================= CONFIG =========================================
DATA_TRAIN = 'data/train.csv'
DATA_TEST  = 'data/test.csv'
SUBMISSION_FILE = 'submission_champion_temporal_minimal.csv'
DIAGNOSTICS_FILE = 'temporal_minimal_diagnostics.json'
RANDOM_STATE = 42
N_CHRONO_SPLITS = 8
ALPHA_GRID = [0.3,0.4,0.5,0.6,0.75,1.0,1.5,2.0,2.5,3.0,4.0]
SEEDS = [42,123,456,789,2024]

np.random.seed(RANDOM_STATE)

# ============================= LOAD DATA ======================================
train_df = pd.read_csv(DATA_TRAIN)
test_df  = pd.read_csv(DATA_TEST)
if 'W' not in train_df.columns:
    raise ValueError('Training data must contain W column')
if 'yearID' not in train_df.columns:
    raise ValueError('Training data must contain yearID for temporal splitting')

y = train_df['W'].astype(float)
years = train_df['yearID'].values

# ============================= FEATURE ENGINEERING ============================

def build_stable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'R','RA','G'}.issubset(df.columns):
        for exp in [1.83,1.9,2.0]:
            e = str(int(exp*100))
            df[f'pyth_exp_{e}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{e}'] = df[f'pyth_exp_{e}'] * df['G']
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / (df['G'] + 1)
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    if 'G' in df.columns:
        for c in ['R','RA','H','HR','BB','SO']:
            if c in df.columns:
                df[f'{c}_per_G'] = df[c] / (df['G'] + 1)
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

train_feat = build_stable(train_df)
test_feat  = build_stable(test_df)
features = sorted((set(train_feat.columns) & set(test_feat.columns)) - EXCLUDE)
X = train_feat[features].values
X_test = test_feat[features].values

scaler = RobustScaler().fit(X)
X_s = scaler.transform(X)
X_test_s = scaler.transform(X_test)

# ============================= CHRONO FOLDS ====================================
# Expanding window chronological splits
sorted_idx = np.argsort(years)
years_sorted = years[sorted_idx]
unique_years = np.unique(years_sorted)
f_year_splits = np.array_split(unique_years, N_CHRONO_SPLITS)
folds = []
for i in range(1, len(f_year_splits)):
    val_years = f_year_splits[i]
    train_years_concat = np.concatenate(f_year_splits[:i])
    tr_mask = np.isin(years, train_years_concat)
    va_mask = np.isin(years, val_years)
    if tr_mask.sum() and va_mask.sum():
        folds.append((tr_mask, va_mask))

# ============================= ALPHA SELECTION =================================
alpha_scores = []
for a in ALPHA_GRID:
    fold_maes = []
    for tr_mask, va_mask in folds:
        m = Ridge(alpha=a)
        m.fit(X_s[tr_mask], y.iloc[tr_mask])
        pred = m.predict(X_s[va_mask])
        fold_maes.append(mean_absolute_error(y.iloc[va_mask], pred))
    alpha_scores.append((a, np.mean(fold_maes)))
alpha_scores.sort(key=lambda x: x[1])
best_alpha, best_chrono_mae = alpha_scores[0]
print(f"Best alpha (chronological): {best_alpha} | MAE={best_chrono_mae:.4f}")

# ============================= OOF & DIAGNOSTICS ===============================
oof = np.zeros(len(y))
for tr_mask, va_mask in folds:
    m = Ridge(alpha=best_alpha)
    m.fit(X_s[tr_mask], y.iloc[tr_mask])
    oof[va_mask] = m.predict(X_s[va_mask])
# Note: earliest fold (if some initial years not in validation) may remain 0 -> mask them
valid_mask = oof != 0
if not valid_mask.all():
    # Replace missing with fitted single-fold prediction to avoid bias in MAE metric
    base_model = Ridge(alpha=best_alpha).fit(X_s[valid_mask], y.iloc[valid_mask])
    oof[~valid_mask] = base_model.predict(X_s[~valid_mask])
oof_mae = mean_absolute_error(y, oof)
print(f"Chronological OOF MAE (recomputed): {oof_mae:.4f}")

# ============================= MULTI-SEED REFIT ================================
seed_preds = []
for sd in SEEDS:
    m = Ridge(alpha=best_alpha, random_state=sd)
    m.fit(X_s, y)
    seed_preds.append(m.predict(X_test_s))
final_float = np.mean(seed_preds, axis=0)

# ============================= FINAL PREDICTIONS ===============================
final_pred = np.clip(final_float, 0, 162).round().astype(int)
submission = pd.DataFrame({'ID': test_df['ID'], 'W': final_pred})
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Saved submission: {SUBMISSION_FILE}")

# ============================= DIAGNOSTICS SAVE ================================
try:
    diag = {
        'best_alpha': best_alpha,
        'chronological_mae': best_chrono_mae,
        'oof_mae': oof_mae,
        'alpha_grid': alpha_scores,
        'prediction_stats': {
            'min': int(final_pred.min()),
            'max': int(final_pred.max()),
            'mean': float(final_pred.mean()),
            'std': float(final_pred.std()),
        },
        'feature_count': len(features),
        'seeds': SEEDS,
        'fold_count': len(folds)
    }
    with open(DIAGNOSTICS_FILE, 'w') as f:
        json.dump(diag, f, indent=2)
    print(f"Diagnostics saved: {DIAGNOSTICS_FILE}")
except Exception as e:
    print('Diagnostics save failed:', e)

# ============================= SUMMARY PRINT ==================================
print('================ TEMPORAL MINIMAL SUMMARY ================')
print(f"Alpha: {best_alpha} | Chrono MAE (selection): {best_chrono_mae:.4f}")
print(f"Recomputed OOF MAE: {oof_mae:.4f}")
print(f"Prediction distribution: min={final_pred.min()} max={final_pred.max()} mean={final_pred.mean():.2f} std={final_pred.std():.2f}")
print(f"Features used: {len(features)} | Folds: {len(folds)} | Seeds: {len(SEEDS)}")
print('==========================================================')
