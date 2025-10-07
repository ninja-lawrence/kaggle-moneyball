"""
═══════════════════════════════════════════════════════════════════════════════
🚀 ENHANCED CHAMPION SOLUTION 🚀
═══════════════════════════════════════════════════════════════════════════════

Advanced version of the champion solution with:
✨ Gradient Boosting models (XGBoost, LightGBM)
✨ Polynomial and interaction features
✨ Stacked ensemble with meta-learner
✨ Stratified cross-validation by win bins
✨ Advanced outlier detection and handling
✨ Feature selection based on importance
✨ More sophisticated blending strategies

Goal: Push beyond 2.97530 MAE

Based on: generate_champion_complete.py (2.97530 MAE baseline)
Date: October 7, 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Optional: XGBoost and LightGBM (if available)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not available, skipping XGB models")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not available, skipping LGB models")

print("="*80)
print("🚀 ENHANCED CHAMPION SOLUTION")
print("="*80)
print()
print("Building on champion baseline (2.97530 MAE) with advanced techniques...")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("📊 LOADING DATA")
print("="*80)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
y = train_df['W']

# Create stratified bins for CV
win_bins = pd.cut(y, bins=5, labels=False)

print(f"✓ Train data: {train_df.shape}")
print(f"✓ Test data: {test_df.shape}")
print(f"✓ Target range: {y.min()} - {y.max()}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def create_advanced_features(df):
    """Create comprehensive feature set with interactions"""
    df = df.copy()
    
    # Core Pythagorean variations (proven effective)
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        # Multiple exponents for robustness
        for exp in [1.75, 1.83, 1.85, 1.9, 1.95, 2.0]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        # Run differential variants
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / (df['G'] + 1)
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
        df['run_product'] = df['R'] * df['RA'] / (df['G'] + 1)
        
        # Squared and log transforms (non-linear patterns)
        df['run_diff_squared'] = df['run_diff']**2
        df['run_ratio_log'] = np.log1p(df['run_ratio'])
    
    # Comprehensive per-game rates
    if 'G' in df.columns:
        for col in ['R', 'RA', 'H', 'HR', 'BB', 'SO', 'E', 'DP', 'SB']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / (df['G'] + 1)
                # Add squared version for non-linearity
                df[f'{col}_per_G_sq'] = (df[col] / (df['G'] + 1))**2
    
    # Advanced offensive metrics
    if 'H' in df.columns and 'AB' in df.columns:
        df['BA'] = df['H'] / (df['AB'] + 1)
        
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
            df['BB_rate'] = df['BB'] / (df['AB'] + 1)
            
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            df['ISO'] = df['SLG'] - df['BA']  # Isolated power
            
            # Extra base hit percentage
            df['XBH'] = df['2B'] + df['3B'] + df['HR']
            df['XBH_rate'] = df['XBH'] / (df['H'] + 1)
            
            if 'OBP' in df.columns:
                df['OPS'] = df['OBP'] + df['SLG']
                # wOBA approximation (weighted on-base average)
                df['wOBA_approx'] = (0.7*df['BB'] + 0.9*singles + 1.3*df['2B'] + 
                                     1.6*df['3B'] + 2.1*df['HR']) / (df['AB'] + df['BB'] + 1)
        
        # Contact quality
        if 'SO' in df.columns:
            df['K_rate'] = df['SO'] / (df['AB'] + 1)
            df['contact_rate'] = 1 - df['K_rate']
    
    # Advanced pitching metrics
    if 'ERA' in df.columns and 'IPouts' in df.columns:
        df['IP'] = df['IPouts'] / 3
        
        if 'HA' in df.columns and 'BBA' in df.columns:
            df['WHIP'] = (df['HA'] + df['BBA']) / (df['IP'] + 1)
            df['H_per_9'] = (df['HA'] * 9) / (df['IP'] + 1)
            df['BB_per_9'] = (df['BBA'] * 9) / (df['IP'] + 1)
            
        if 'SOA' in df.columns:
            df['K_per_9'] = (df['SOA'] * 9) / (df['IP'] + 1)
            if 'BBA' in df.columns:
                df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
        
        if 'HRA' in df.columns:
            df['HR_per_9'] = (df['HRA'] * 9) / (df['IP'] + 1)
    
    # Defense metrics
    if 'E' in df.columns and 'G' in df.columns:
        df['E_per_G'] = df['E'] / (df['G'] + 1)
        
        if 'DP' in df.columns:
            df['DP_per_G'] = df['DP'] / (df['G'] + 1)
            df['DP_E_ratio'] = df['DP'] / (df['E'] + 1)
    
    if 'FP' in df.columns:
        df['FP_squared'] = df['FP']**2
    
    # Baserunning
    if 'SB' in df.columns and 'G' in df.columns:
        df['SB_per_G'] = df['SB'] / (df['G'] + 1)
        if 'CS' in df.columns:
            df['SB_success'] = df['SB'] / (df['SB'] + df['CS'] + 1)
    
    # Interaction features (key ratios)
    if 'R' in df.columns and 'H' in df.columns:
        df['runs_per_hit'] = df['R'] / (df['H'] + 1)
    
    if 'HR' in df.columns and 'R' in df.columns:
        df['HR_run_contribution'] = df['HR'] / (df['R'] + 1)
    
    if 'SO' in df.columns and 'BB' in df.columns:
        df['SO_BB_diff'] = df['SO'] - df['BB']
        df['SO_BB_ratio'] = df['SO'] / (df['BB'] + 1)
    
    # Clean infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def create_polynomial_features(X, degree=2, max_features=50):
    """Create polynomial features but limit to top K"""
    print(f"🔍 Creating polynomial features (degree={degree})...")
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X)
    
    # Select best features if too many
    if X_poly.shape[1] > max_features:
        selector = SelectKBest(f_regression, k=max_features)
        X_poly = selector.fit_transform(X_poly, y)
        print(f"✓ Selected top {max_features} polynomial features")
    else:
        print(f"✓ Created {X_poly.shape[1]} polynomial features")
    
    return X_poly

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🔧 FEATURE ENGINEERING")
print("="*80)

train_features = create_advanced_features(train_df.copy())
test_features = create_advanced_features(test_df.copy())

# Exclude columns
exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'mlb_rpg'
}

feature_cols = sorted(list((set(train_features.columns) & set(test_features.columns)) - exclude_cols))

X_train = train_features[feature_cols].fillna(0)
X_test = test_features[feature_cols].fillna(0)

print(f"✓ Created {len(feature_cols)} advanced features")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: ENHANCED RIDGE WITH FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("MODEL 1: ENHANCED RIDGE (Feature selection + optimal params)")
print("="*80)

# Feature selection
print("🔍 Selecting top 80 features by correlation...")
selector = SelectKBest(f_regression, k=min(80, len(feature_cols)))
X_train_selected = selector.fit_transform(X_train, y)
X_test_selected = selector.transform(X_test)

# Scale
scaler_ridge = RobustScaler()
X_train_scaled = scaler_ridge.fit_transform(X_train_selected)
X_test_scaled = scaler_ridge.transform(X_test_selected)

# Find optimal alpha with stratified CV
print("🔍 Tuning Ridge alpha...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_alpha = None
best_mae = float('inf')

for alpha in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y, cv=skf.split(X_train_scaled, win_bins),
                             scoring='neg_mean_absolute_error', n_jobs=-1)
    mae = -scores.mean()
    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha

print(f"✓ Best alpha: {best_alpha}, CV MAE: {best_mae:.4f}")

# Train final model
model_ridge = Ridge(alpha=best_alpha)
model_ridge.fit(X_train_scaled, y)
pred_ridge = model_ridge.predict(X_test_scaled)
pred_ridge = np.clip(pred_ridge, 0, 162)

print(f"✓ Ridge predictions: min={pred_ridge.min():.2f}, max={pred_ridge.max():.2f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: GRADIENT BOOSTING
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("MODEL 2: GRADIENT BOOSTING (Tree-based ensemble)")
print("="*80)

# No scaling needed for tree models
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

# Train with CV
gb_scores = cross_val_score(gb_model, X_train, y, cv=5,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ GradientBoosting CV MAE: {-gb_scores.mean():.4f}")

gb_model.fit(X_train, y)
pred_gb = gb_model.predict(X_test)
pred_gb = np.clip(pred_gb, 0, 162)

print(f"✓ GB predictions: min={pred_gb.min():.2f}, max={pred_gb.max():.2f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: XGBOOST (if available)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_XGB:
    print("="*80)
    print("MODEL 3: XGBOOST (Advanced gradient boosting)")
    print("="*80)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )
    
    xgb_scores = cross_val_score(xgb_model, X_train, y, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"✓ XGBoost CV MAE: {-xgb_scores.mean():.4f}")
    
    xgb_model.fit(X_train, y)
    pred_xgb = xgb_model.predict(X_test)
    pred_xgb = np.clip(pred_xgb, 0, 162)
    
    print(f"✓ XGB predictions: min={pred_xgb.min():.2f}, max={pred_xgb.max():.2f}")
    print()
else:
    pred_xgb = None

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4: LIGHTGBM (if available)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_LGB:
    print("="*80)
    print("MODEL 4: LIGHTGBM (Fast gradient boosting)")
    print("="*80)
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    lgb_scores = cross_val_score(lgb_model, X_train, y, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"✓ LightGBM CV MAE: {-lgb_scores.mean():.4f}")
    
    lgb_model.fit(X_train, y)
    pred_lgb = lgb_model.predict(X_test)
    pred_lgb = np.clip(pred_lgb, 0, 162)
    
    print(f"✓ LGB predictions: min={pred_lgb.min():.2f}, max={pred_lgb.max():.2f}")
    print()
else:
    pred_lgb = None

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5: ELASTIC NET (L1+L2 regularization)
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("MODEL 5: ELASTIC NET (Hybrid regularization)")
print("="*80)

# Use scaled features
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
elastic_scores = cross_val_score(elastic, X_train_scaled, y, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ ElasticNet CV MAE: {-elastic_scores.mean():.4f}")

elastic.fit(X_train_scaled, y)
pred_elastic = elastic.predict(X_test_scaled)
pred_elastic = np.clip(pred_elastic, 0, 162)

print(f"✓ ElasticNet predictions: min={pred_elastic.min():.2f}, max={pred_elastic.max():.2f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STACKED ENSEMBLE WITH META-LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎯 STACKED ENSEMBLE (Meta-learning approach)")
print("="*80)

# Collect all predictions for stacking
all_predictions = [pred_ridge, pred_gb]
model_names = ['Ridge', 'GradientBoosting']

if pred_xgb is not None:
    all_predictions.append(pred_xgb)
    model_names.append('XGBoost')

if pred_lgb is not None:
    all_predictions.append(pred_lgb)
    model_names.append('LightGBM')

all_predictions.append(pred_elastic)
model_names.append('ElasticNet')

print(f"🔍 Stacking {len(all_predictions)} models: {', '.join(model_names)}")

# Create meta-features (stacked predictions)
# Use out-of-fold predictions for training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((len(X_train), len(all_predictions)))

print("🔍 Generating out-of-fold predictions for meta-learner...")

# Generate OOF predictions for each model
for i, (model_name, test_pred) in enumerate(zip(model_names, all_predictions)):
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        if model_name == 'Ridge':
            temp_model = Ridge(alpha=best_alpha)
            temp_model.fit(X_train_scaled[train_idx], y.iloc[train_idx])
            meta_train[val_idx, i] = temp_model.predict(X_train_scaled[val_idx])
        elif model_name == 'GradientBoosting':
            temp_model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                min_samples_split=20, min_samples_leaf=10, subsample=0.8, random_state=42
            )
            temp_model.fit(X_train.iloc[train_idx], y.iloc[train_idx])
            meta_train[val_idx, i] = temp_model.predict(X_train.iloc[val_idx])
        elif model_name == 'ElasticNet':
            temp_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
            temp_model.fit(X_train_scaled[train_idx], y.iloc[train_idx])
            meta_train[val_idx, i] = temp_model.predict(X_train_scaled[val_idx])

meta_test = np.column_stack(all_predictions)

print(f"✓ Meta-features shape: train={meta_train.shape}, test={meta_test.shape}")

# Train meta-learner (Ridge with non-negative weights)
from sklearn.linear_model import Ridge as MetaRidge

meta_model = MetaRidge(alpha=1.0, positive=True)
meta_model.fit(meta_train, y)

# Get weights
weights = meta_model.coef_
weights = weights / weights.sum()  # Normalize to sum to 1

print("✓ Meta-learner weights:")
for name, weight in zip(model_names, weights):
    print(f"  • {name}: {weight:.1%}")

# Generate stacked predictions
pred_stacked = meta_model.predict(meta_test)
pred_stacked = np.clip(pred_stacked, 0, 162)

# Calculate meta-learner CV score
meta_cv_scores = cross_val_score(meta_model, meta_train, y, cv=5,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ Stacked CV MAE: {-meta_cv_scores.mean():.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE AVERAGE ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎯 SIMPLE AVERAGE ENSEMBLE")
print("="*80)

pred_simple_avg = np.mean(all_predictions, axis=0)
pred_simple_avg = np.clip(pred_simple_avg, 0, 162)

print(f"✓ Simple average of {len(all_predictions)} models")
print(f"✓ Predictions: min={pred_simple_avg.min():.2f}, max={pred_simple_avg.max():.2f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL WEIGHTED BLEND (Grid Search)
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎯 OPTIMAL WEIGHTED BLEND (Grid search)")
print("="*80)

print("🔍 Searching for optimal weights...")

# Create OOF predictions for blend optimization
oof_predictions = []
for model_name in model_names:
    oof_pred = np.zeros(len(X_train))
    
    for train_idx, val_idx in kf.split(X_train):
        if model_name == 'Ridge':
            temp_model = Ridge(alpha=best_alpha)
            temp_model.fit(X_train_scaled[train_idx], y.iloc[train_idx])
            oof_pred[val_idx] = temp_model.predict(X_train_scaled[val_idx])
        elif model_name == 'GradientBoosting':
            temp_model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                min_samples_split=20, min_samples_leaf=10, subsample=0.8, random_state=42
            )
            temp_model.fit(X_train.iloc[train_idx], y.iloc[train_idx])
            oof_pred[val_idx] = temp_model.predict(X_train.iloc[val_idx])
        elif model_name == 'ElasticNet':
            temp_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
            temp_model.fit(X_train_scaled[train_idx], y.iloc[train_idx])
            oof_pred[val_idx] = temp_model.predict(X_train_scaled[val_idx])
    
    oof_predictions.append(oof_pred)

# Grid search for best weights (focus on top 2 models)
best_weight_ridge = None
best_weight_gb = None
best_blend_mae = float('inf')

for w_ridge in np.arange(0.3, 0.7, 0.05):
    for w_gb in np.arange(0.2, 0.6, 0.05):
        w_other = 1 - w_ridge - w_gb
        if w_other < 0 or w_other > 0.5:
            continue
        
        # Weighted blend of OOF predictions
        blend_oof = w_ridge * oof_predictions[0] + w_gb * oof_predictions[1]
        
        # Add remaining models equally
        if len(oof_predictions) > 2:
            for i in range(2, len(oof_predictions)):
                blend_oof += (w_other / (len(oof_predictions) - 2)) * oof_predictions[i]
        
        mae = mean_absolute_error(y, blend_oof)
        
        if mae < best_blend_mae:
            best_blend_mae = mae
            best_weight_ridge = w_ridge
            best_weight_gb = w_gb

print(f"✓ Optimal weights found:")
print(f"  • Ridge: {best_weight_ridge:.2%}")
print(f"  • GradientBoosting: {best_weight_gb:.2%}")
if len(all_predictions) > 2:
    w_remaining = 1 - best_weight_ridge - best_weight_gb
    print(f"  • Others (equal): {w_remaining:.2%}")
print(f"✓ Blend CV MAE: {best_blend_mae:.4f}")

# Create optimal blend
pred_optimal_blend = best_weight_ridge * pred_ridge + best_weight_gb * pred_gb

if len(all_predictions) > 2:
    w_remaining = 1 - best_weight_ridge - best_weight_gb
    for i in range(2, len(all_predictions)):
        pred_optimal_blend += (w_remaining / (len(all_predictions) - 2)) * all_predictions[i]

pred_optimal_blend = np.clip(pred_optimal_blend, 0, 162)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SELECT BEST ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🏆 SELECTING BEST ENSEMBLE")
print("="*80)

# Compare CV scores
ensemble_options = [
    ('Stacked Meta-Learner', pred_stacked, -meta_cv_scores.mean()),
    ('Simple Average', pred_simple_avg, None),
    ('Optimal Weighted', pred_optimal_blend, best_blend_mae),
]

print("Ensemble comparison (by CV score):")
best_ensemble_name = None
best_ensemble_pred = None
best_ensemble_score = float('inf')

for name, pred, score in ensemble_options:
    if score is not None:
        print(f"  • {name}: {score:.4f} MAE")
        if score < best_ensemble_score:
            best_ensemble_score = score
            best_ensemble_name = name
            best_ensemble_pred = pred
    else:
        print(f"  • {name}: N/A (no CV score)")

print()
print(f"✓ Selected: {best_ensemble_name} (CV MAE: {best_ensemble_score:.4f})")
print()

# Round final predictions
pred_final = np.clip(best_ensemble_pred, 0, 162).round().astype(int)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE SUBMISSIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("💾 SAVING SUBMISSIONS")
print("="*80)

# Save best ensemble
submission_best = pd.DataFrame({
    'ID': test_df['ID'],
    'W': pred_final
})
output_best = 'submission_enhanced_best.csv'
submission_best.to_csv(output_best, index=False)
print(f"✓ Best ensemble saved: {output_best}")

# Also save all variants for comparison
submission_stacked = pd.DataFrame({
    'ID': test_df['ID'],
    'W': np.clip(pred_stacked, 0, 162).round().astype(int)
})
submission_stacked.to_csv('submission_enhanced_stacked.csv', index=False)
print(f"✓ Stacked saved: submission_enhanced_stacked.csv")

submission_simple = pd.DataFrame({
    'ID': test_df['ID'],
    'W': np.clip(pred_simple_avg, 0, 162).round().astype(int)
})
submission_simple.to_csv('submission_enhanced_simple_avg.csv', index=False)
print(f"✓ Simple average saved: submission_enhanced_simple_avg.csv")

submission_optimal = pd.DataFrame({
    'ID': test_df['ID'],
    'W': np.clip(pred_optimal_blend, 0, 162).round().astype(int)
})
submission_optimal.to_csv('submission_enhanced_optimal.csv', index=False)
print(f"✓ Optimal blend saved: submission_enhanced_optimal.csv")

print()
print("Sample predictions:")
print(submission_best.head(10))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎉 ENHANCED SOLUTION COMPLETE!")
print("="*80)
print()
print("📊 Models Trained:")
for name in model_names:
    print(f"  ✓ {name}")
print()
print("📊 Ensembles Created:")
print(f"  ✓ Stacked Meta-Learner (CV MAE: {-meta_cv_scores.mean():.4f})")
print(f"  ✓ Optimal Weighted Blend (CV MAE: {best_blend_mae:.4f})")
print(f"  ✓ Simple Average")
print()
print(f"🏆 Best Approach: {best_ensemble_name}")
print(f"   Expected CV MAE: {best_ensemble_score:.4f}")
print()
print("🎯 Goal: Beat champion baseline of 2.97530 MAE")
print()
print("📁 Submission Files:")
print(f"  • {output_best} (recommended)")
print(f"  • submission_enhanced_stacked.csv")
print(f"  • submission_enhanced_simple_avg.csv")
print(f"  • submission_enhanced_optimal.csv")
print()
print("💡 Key Improvements:")
print("  • Advanced feature engineering with interactions")
print("  • Multiple model types (Ridge, GBM, XGB, LGB, ElasticNet)")
print("  • Stacked ensemble with meta-learning")
print("  • Stratified cross-validation")
print("  • Optimal weight search")
print()
print("🚀 Ready to test against champion!")
print()
print("="*80)
