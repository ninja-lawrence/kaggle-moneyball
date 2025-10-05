import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================================
# CRITICAL INSIGHT: Focus on pythagorean expectation + key adjustments
# ============================================================================

def create_features(df):
    """Create only the most stable, generalizable features"""
    df = df.copy()
    
    # Pythagorean expectation variations (MOST IMPORTANT)
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        # Standard (exponent=2)
        df['pyth_exp'] = df['R']**2 / (df['R']**2 + df['RA']**2 + 1)
        df['pyth_wins'] = df['pyth_exp'] * df['G']
        
        # Optimized (exponent=1.83) - often better
        df['pyth_exp_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83 + 1)
        df['pyth_wins_183'] = df['pyth_exp_183'] * df['G']
        
        # Alternative exponents
        df['pyth_exp_185'] = df['R']**1.85 / (df['R']**1.85 + df['RA']**1.85 + 1)
        df['pyth_wins_185'] = df['pyth_exp_185'] * df['G']
        
        df['pyth_exp_19'] = df['R']**1.9 / (df['R']**1.9 + df['RA']**1.9 + 1)
        df['pyth_wins_19'] = df['pyth_exp_19'] * df['G']
    
    # Run differential and variations
    if 'R' in df.columns and 'RA' in df.columns:
        df['run_diff'] = df['R'] - df['RA']
        if 'G' in df.columns:
            df['run_diff_per_game'] = df['run_diff'] / df['G']
            
        # Non-linear transformations
        df['run_diff_sqrt'] = np.sign(df['run_diff']) * np.sqrt(np.abs(df['run_diff']))
        df['run_diff_sq'] = df['run_diff']**2
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    
    # Basic rates
    if 'G' in df.columns:
        for col in ['R', 'RA', 'H', 'HR', 'BB', 'SO']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / df['G']
    
    # Offensive metrics
    if 'H' in df.columns and 'AB' in df.columns:
        df['BA'] = df['H'] / (df['AB'] + 1)
        
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            
            if 'OBP' in df.columns:
                df['OPS'] = df['OBP'] + df['SLG']
    
    # Pitching efficiency
    if 'ERA' in df.columns and 'IPouts' in df.columns:
        if 'HA' in df.columns and 'BBA' in df.columns:
            df['WHIP'] = (df['HA'] + df['BBA']) / ((df['IPouts'] / 3) + 1)
            
        if 'SOA' in df.columns:
            df['K_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 1)
    
    # Context adjustment
    if 'mlb_rpg' in df.columns and 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        df['R_vs_league'] = (df['R'] / df['G']) - df['mlb_rpg']
        df['RA_vs_league'] = (df['RA'] / df['G']) - df['mlb_rpg']
        df['net_vs_league'] = df['R_vs_league'] - df['RA_vs_league']
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

print("\nCreating features...")
train_df = create_features(train_df)
test_df = create_features(test_df)

# Get common features
exclude_cols = {'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins'}
train_features = set(train_df.columns) - exclude_cols
test_features = set(test_df.columns) - exclude_cols
common_features = sorted(list(train_features & test_features))

print(f"Using {len(common_features)} features")

# Prepare data
X_train = train_df[common_features].fillna(0)
y_train = train_df['W']
X_test = test_df[common_features].fillna(0)
test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df.index

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nX_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")

# ============================================================================
# RIDGE REGRESSION
# ============================================================================
print("\n" + "="*80)
print("RIDGE REGRESSION")
print("="*80)

# Quick Ridge with optimal alpha from previous runs
alphas = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
best_ridge_alpha = None
best_ridge_cv = float('inf')

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=10,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    if cv_mae < best_ridge_cv:
        best_ridge_cv = cv_mae
        best_ridge_alpha = alpha

print(f"Best Ridge alpha: {best_ridge_alpha}, CV MAE: {best_ridge_cv:.4f}")

# Train Ridge model
ridge_model = Ridge(alpha=best_ridge_alpha)
ridge_model.fit(X_train_scaled, y_train)
ridge_train_pred = ridge_model.predict(X_train_scaled)
ridge_test_pred = ridge_model.predict(X_test_scaled)

# ============================================================================
# XGBOOST WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("XGBOOST HYPERPARAMETER TUNING")
print("="*80)

# XGBoost hyperparameter grid
param_grid = [
    # Conservative models (prevent overfitting)
    {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3},
    {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3},
    {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 250, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3},
    # Slightly more complex
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 2},
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 2},
]

best_params = None
best_cv_mae = float('inf')
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for i, params in enumerate(param_grid, 1):
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1,
        **params
    )
    
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=kfold,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Config {i}: depth={params['max_depth']}, lr={params['learning_rate']}, "
          f"n_est={params['n_estimators']}, MAE={cv_mae:.4f}±{cv_std:.4f}")
    
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_params = params

print(f"\nBest CV MAE: {best_cv_mae:.4f}")
print(f"Best params: {best_params}")

# ============================================================================
# FINAL XGBOOST MODEL
# ============================================================================
print("\n" + "="*80)
print("TRAINING FINAL XGBOOST MODEL")
print("="*80)

final_model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    random_state=42,
    n_jobs=-1,
    **best_params
)
final_model.fit(X_train, y_train)

# Get XGBoost predictions
xgb_train_pred = final_model.predict(X_train)
xgb_test_pred = final_model.predict(X_test)

# XGBoost Metrics
xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
print(f"XGBoost Train MAE: {xgb_train_mae:.4f}")
print(f"XGBoost CV MAE: {best_cv_mae:.4f}")

# ============================================================================
# ENSEMBLE: BLEND RIDGE AND XGBOOST
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE MODEL (Ridge + XGBoost)")
print("="*80)

# Try different blending weights
best_blend_weight = None
best_blend_cv = float('inf')

# Use cross-validation to find best blend
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for ridge_weight in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    xgb_weight = 1 - ridge_weight
    blend_maes = []
    
    for train_idx, val_idx in kfold.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        X_tr_scaled, X_val_scaled = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train both models on fold
        ridge_fold = Ridge(alpha=best_ridge_alpha)
        ridge_fold.fit(X_tr_scaled, y_tr)
        ridge_val_pred = ridge_fold.predict(X_val_scaled)
        
        xgb_fold = xgb.XGBRegressor(objective='reg:absoluteerror', random_state=42, n_jobs=-1, **best_params)
        xgb_fold.fit(X_tr, y_tr)
        xgb_val_pred = xgb_fold.predict(X_val)
        
        # Blend predictions
        blend_pred = ridge_weight * ridge_val_pred + xgb_weight * xgb_val_pred
        blend_mae = mean_absolute_error(y_val, blend_pred)
        blend_maes.append(blend_mae)
    
    avg_blend_mae = np.mean(blend_maes)
    print(f"Ridge weight: {ridge_weight:.1f}, XGB weight: {xgb_weight:.1f}, CV MAE: {avg_blend_mae:.4f}")
    
    if avg_blend_mae < best_blend_cv:
        best_blend_cv = avg_blend_mae
        best_blend_weight = ridge_weight

print(f"\nBest blend: Ridge={best_blend_weight:.1f}, XGBoost={1-best_blend_weight:.1f}")
print(f"Ensemble CV MAE: {best_blend_cv:.4f}")
print(f"Ridge CV MAE: {best_ridge_cv:.4f}")
print(f"XGBoost CV MAE: {best_cv_mae:.4f}")

# Create final ensemble predictions
ensemble_test_pred = best_blend_weight * ridge_test_pred + (1 - best_blend_weight) * xgb_test_pred

# Clip and create submission
ensemble_test_pred = np.clip(ensemble_test_pred, 0, 162)
ensemble_test_pred_int = np.round(ensemble_test_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': ensemble_test_pred_int
})

submission_df.to_csv('submission_ensemble.csv', index=False)

print("\n" + "="*80)
print("ENSEMBLE SUBMISSION CREATED")
print("="*80)
print(f"File: submission_ensemble.csv")
print(f"Expected Kaggle MAE: ~{best_blend_cv:.2f}")
print(f"\nFirst 5 rows:")
print(submission_df.head())
print(f"\nLast 5 rows:")
print(submission_df.tail())
print(f"\nPrediction stats:")
print(f"  Mean: {ensemble_test_pred_int.mean():.2f}")
print(f"  Std: {ensemble_test_pred_int.std():.2f}")
print(f"  Range: {ensemble_test_pred_int.min()} to {ensemble_test_pred_int.max()}")

# Feature importance (gain-based)
print("\n" + "="*80)
print("TOP 20 FEATURES (by XGBoost importance)")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': common_features,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

print("\n" + "="*80)
print("ENSEMBLE STRATEGY")
print("="*80)
print("Ridge strengths: Linear relationship, stable across eras")
print("XGBoost strengths: Non-linear patterns, feature interactions")
print("Ensemble: Combines both approaches for better generalization")
print(f"\nModel Comparison:")
print(f"  Ridge CV MAE:    {best_ridge_cv:.4f}")
print(f"  XGBoost CV MAE:  {best_cv_mae:.4f}")
print(f"  Ensemble CV MAE: {best_blend_cv:.4f}")
print(f"\nExpected Kaggle: ~{best_blend_cv:.2f}-{best_blend_cv+0.2:.2f}")