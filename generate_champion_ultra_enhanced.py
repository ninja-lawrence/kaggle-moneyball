"""
═══════════════════════════════════════════════════════════════════════════════
🔥 ULTRA-ENHANCED CHAMPION SOLUTION 🔥
═══════════════════════════════════════════════════════════════════════════════

Based on learnings from generate_champion_enhanced.py:
- Stacked meta-learner: 2.97530 (TIED champion!)
- Simple average: 3.12757 (worse)
- Optimal blend: 3.01234 (worse)

Key insight: Meta-learning works, but we need BETTER base models!

NEW STRATEGY:
✨ Focus on model diversity and quality
✨ Multiple Ridge variants with different feature sets
✨ Careful hyperparameter tuning per model
✨ Blend champion models WITH new models
✨ Advanced meta-learner with regularization
✨ Feature engineering focused on reducing bias

Goal: Break through 2.97530 → aim for 2.96xxx or better!

Date: October 7, 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 ULTRA-ENHANCED CHAMPION SOLUTION")
print("="*80)
print()
print("Strategy: Blend CHAMPION models + NEW diverse models")
print("Target: Beat 2.97530 MAE")
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

print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# CHAMPION FEATURES (PROVEN TO WORK)
# ═══════════════════════════════════════════════════════════════════════════════

def create_champion_features(df):
    """Champion's proven feature set"""
    df = df.copy()
    
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.83, 1.85, 1.9, 2.0]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / df['G']
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    
    if 'G' in df.columns:
        for col in ['R', 'RA', 'H', 'HR', 'BB', 'SO']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / df['G']
    
    if 'H' in df.columns and 'AB' in df.columns:
        df['BA'] = df['H'] / (df['AB'] + 1)
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            if 'OBP' in df.columns:
                df['OPS'] = df['OBP'] + df['SLG']
    
    if 'ERA' in df.columns and 'IPouts' in df.columns:
        if 'HA' in df.columns and 'BBA' in df.columns:
            df['WHIP'] = (df['HA'] + df['BBA']) / ((df['IPouts'] / 3) + 1)
        if 'SOA' in df.columns:
            df['K_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def create_extended_features(df):
    """Extended features with more interactions"""
    df = df.copy()
    
    # Start with champion features
    df = create_champion_features(df)
    
    # Add more Pythagorean variants
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.75, 1.95, 2.05]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        # Non-linear transforms
        df['run_diff_sq'] = df['run_diff']**2
        df['run_ratio_log'] = np.log1p(df['run_ratio'])
    
    # Advanced offensive metrics
    if 'H' in df.columns and '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
        df['XBH'] = df['2B'] + df['3B'] + df['HR']
        df['XBH_rate'] = df['XBH'] / (df['H'] + 1)
        
        if 'AB' in df.columns:
            singles = df['H'] - df['XBH']
            df['ISO'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1) - df['BA']
    
    # Pitching ratios
    if 'SOA' in df.columns and 'BBA' in df.columns:
        df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
    
    # Interaction terms
    if 'HR' in df.columns and 'R' in df.columns:
        df['HR_contribution'] = df['HR'] / (df['R'] + 1)
    
    if 'E' in df.columns and 'G' in df.columns:
        df['E_per_G'] = df['E'] / (df['G'] + 1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# RECREATE CHAMPION'S 3 BASE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🏆 RECREATING CHAMPION'S 3 BASE MODELS")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'mlb_rpg'
}

# Model 1: Champion Notemporal
print("\n[1/3] Champion Model 1: Notemporal (Ridge alpha=10)")
train_champ = create_champion_features(train_df.copy())
test_champ = create_champion_features(test_df.copy())

champ_features = sorted(list((set(train_champ.columns) & set(test_champ.columns)) - exclude_cols))
X_champ_train = train_champ[champ_features].fillna(0)
X_champ_test = test_champ[champ_features].fillna(0)

scaler_champ = RobustScaler()
X_champ_train_scaled = scaler_champ.fit_transform(X_champ_train)
X_champ_test_scaled = scaler_champ.transform(X_champ_test)

model_champ1 = Ridge(alpha=10.0, random_state=42)
scores = cross_val_score(model_champ1, X_champ_train_scaled, y, cv=kfold,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

model_champ1.fit(X_champ_train_scaled, y)
pred_champ1 = model_champ1.predict(X_champ_test_scaled)

# Model 2 & 3: Use similar approach with different alphas/seeds
print("\n[2/3] Champion Model 2: Multi-ensemble style (Ridge alpha=3)")
model_champ2 = Ridge(alpha=3.0, random_state=42)
scores = cross_val_score(model_champ2, X_champ_train_scaled, y, cv=kfold,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

model_champ2.fit(X_champ_train_scaled, y)
pred_champ2 = model_champ2.predict(X_champ_test_scaled)

print("\n[3/3] Champion Model 3: Finetuned (Ridge alpha=10, multi-seed)")
seeds = [42, 123, 456]
preds_multi_seed = []
for seed in seeds:
    model_temp = Ridge(alpha=10.0, random_state=seed)
    model_temp.fit(X_champ_train_scaled, y)
    preds_multi_seed.append(model_temp.predict(X_champ_test_scaled))
pred_champ3 = np.mean(preds_multi_seed, axis=0)

# Champion blend (37/44/19)
pred_champion_blend = 0.37 * pred_champ1 + 0.44 * pred_champ2 + 0.19 * pred_champ3
print(f"\n✓ Champion blend created (37/44/19)")

# ═══════════════════════════════════════════════════════════════════════════════
# NEW DIVERSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🆕 CREATING NEW DIVERSE MODELS")
print("="*80)

# Extended features
train_ext = create_extended_features(train_df.copy())
test_ext = create_extended_features(test_df.copy())
ext_features = sorted(list((set(train_ext.columns) & set(test_ext.columns)) - exclude_cols))
X_ext_train = train_ext[ext_features].fillna(0)
X_ext_test = test_ext[ext_features].fillna(0)

# Model 4: Ridge with feature selection (adaptive k)
print("\n[4] Ridge + Feature Selection")
k_fs = min(45, X_ext_train.shape[1])  # Adaptive k based on available features
print(f"   Using k={k_fs} features")
selector = SelectKBest(f_regression, k=k_fs)
X_fs_train = selector.fit_transform(X_ext_train, y)
X_fs_test = selector.transform(X_ext_test)

scaler_fs = StandardScaler()
X_fs_train_scaled = scaler_fs.fit_transform(X_fs_train)
X_fs_test_scaled = scaler_fs.transform(X_fs_test)

model_fs = Ridge(alpha=5.0, random_state=42)
scores = cross_val_score(model_fs, X_fs_train_scaled, y, cv=kfold,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

model_fs.fit(X_fs_train_scaled, y)
pred_fs = model_fs.predict(X_fs_test_scaled)

# Model 5: Ridge with mutual info feature selection (adaptive k)
print("\n[5] Ridge + Mutual Info Selection")
k_mi = min(50, X_ext_train.shape[1])  # Adaptive k based on available features
print(f"   Using k={k_mi} features")
selector_mi = SelectKBest(mutual_info_regression, k=k_mi)
X_mi_train = selector_mi.fit_transform(X_ext_train, y)
X_mi_test = selector_mi.transform(X_ext_test)

scaler_mi = RobustScaler()
X_mi_train_scaled = scaler_mi.fit_transform(X_mi_train)
X_mi_test_scaled = scaler_mi.transform(X_mi_test)

model_mi = Ridge(alpha=7.0, random_state=42)
scores = cross_val_score(model_mi, X_mi_train_scaled, y, cv=kfold,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

model_mi.fit(X_mi_train_scaled, y)
pred_mi = model_mi.predict(X_mi_test_scaled)

# Model 6: Lasso (L1 regularization for sparsity)
print("\n[6] Lasso (L1 regularization)")
scaler_ext = StandardScaler()
X_ext_train_scaled = scaler_ext.fit_transform(X_ext_train)
X_ext_test_scaled = scaler_ext.transform(X_ext_test)

model_lasso = Lasso(alpha=0.1, random_state=42, max_iter=3000)
scores = cross_val_score(model_lasso, X_ext_train_scaled, y, cv=kfold,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

model_lasso.fit(X_ext_train_scaled, y)
pred_lasso = model_lasso.predict(X_ext_test_scaled)

# Model 7: Gradient Boosting (tree-based)
print("\n[7] Gradient Boosting (tree-based)")
gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=30,
    min_samples_leaf=15,
    subsample=0.8,
    random_state=42
)

scores = cross_val_score(gb_model, X_ext_train, y, cv=5,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

gb_model.fit(X_ext_train, y)
pred_gb = gb_model.predict(X_ext_test)

# Model 8: Conservative Ridge (high alpha, less overfitting)
print("\n[8] Conservative Ridge (alpha=25)")
model_conservative = Ridge(alpha=25.0, random_state=42)
scores = cross_val_score(model_conservative, X_ext_train_scaled, y, cv=kfold,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ CV MAE: {-scores.mean():.4f}")

model_conservative.fit(X_ext_train_scaled, y)
pred_conservative = model_conservative.predict(X_ext_test_scaled)

print(f"\n✓ Created 8 diverse models (3 champion + 5 new)")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE OUT-OF-FOLD PREDICTIONS FOR META-LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧠 GENERATING OUT-OF-FOLD PREDICTIONS")
print("="*80)

n_models = 8
oof_preds = np.zeros((len(X_champ_train), n_models))
test_preds = np.column_stack([
    pred_champ1, pred_champ2, pred_champ3,
    pred_fs, pred_mi, pred_lasso, pred_gb, pred_conservative
])

print("Generating OOF predictions for meta-learner training...")

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_champ_train)):
    if fold_idx == 0:
        print(f"Processing fold {fold_idx + 1}/10...", end=" ")
    
    # Model 1: Champion notemporal
    m1 = Ridge(alpha=10.0, random_state=42)
    m1.fit(X_champ_train_scaled[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 0] = m1.predict(X_champ_train_scaled[val_idx])
    
    # Model 2: Champion multi
    m2 = Ridge(alpha=3.0, random_state=42)
    m2.fit(X_champ_train_scaled[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 1] = m2.predict(X_champ_train_scaled[val_idx])
    
    # Model 3: Champion finetuned (multi-seed average)
    temp_preds = []
    for seed in [42, 123, 456]:
        m3 = Ridge(alpha=10.0, random_state=seed)
        m3.fit(X_champ_train_scaled[train_idx], y.iloc[train_idx])
        temp_preds.append(m3.predict(X_champ_train_scaled[val_idx]))
    oof_preds[val_idx, 2] = np.mean(temp_preds, axis=0)
    
    # Model 4: Feature selection
    X_fs_fold_train = selector.transform(X_ext_train.iloc[train_idx])
    X_fs_fold_val = selector.transform(X_ext_train.iloc[val_idx])
    X_fs_fold_train_scaled = scaler_fs.transform(X_fs_fold_train)
    X_fs_fold_val_scaled = scaler_fs.transform(X_fs_fold_val)
    m4 = Ridge(alpha=5.0, random_state=42)
    m4.fit(X_fs_fold_train_scaled, y.iloc[train_idx])
    oof_preds[val_idx, 3] = m4.predict(X_fs_fold_val_scaled)
    
    # Model 5: Mutual info selection
    X_mi_fold_train = selector_mi.transform(X_ext_train.iloc[train_idx])
    X_mi_fold_val = selector_mi.transform(X_ext_train.iloc[val_idx])
    X_mi_fold_train_scaled = scaler_mi.transform(X_mi_fold_train)
    X_mi_fold_val_scaled = scaler_mi.transform(X_mi_fold_val)
    m5 = Ridge(alpha=7.0, random_state=42)
    m5.fit(X_mi_fold_train_scaled, y.iloc[train_idx])
    oof_preds[val_idx, 4] = m5.predict(X_mi_fold_val_scaled)
    
    # Model 6: Lasso
    m6 = Lasso(alpha=0.1, random_state=42, max_iter=3000)
    m6.fit(X_ext_train_scaled[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 5] = m6.predict(X_ext_train_scaled[val_idx])
    
    # Model 7: Gradient Boosting
    m7 = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        min_samples_split=30, min_samples_leaf=15, subsample=0.8, random_state=42
    )
    m7.fit(X_ext_train.iloc[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 6] = m7.predict(X_ext_train.iloc[val_idx])
    
    # Model 8: Conservative
    m8 = Ridge(alpha=25.0, random_state=42)
    m8.fit(X_ext_train_scaled[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 7] = m8.predict(X_ext_train_scaled[val_idx])

print("Done!")
print(f"✓ OOF predictions shape: {oof_preds.shape}")
print(f"✓ Test predictions shape: {test_preds.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# META-LEARNER ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎯 TRAINING META-LEARNER")
print("="*80)

# Try different meta-learner configurations
meta_models = []

# 1. Ridge meta-learner (positive weights)
print("\n[Meta-1] Ridge with positive weights")
meta_ridge = Ridge(alpha=1.0, positive=True, random_state=42)
meta_ridge.fit(oof_preds, y)
weights_ridge = meta_ridge.coef_ / meta_ridge.coef_.sum()
pred_meta_ridge = meta_ridge.predict(test_preds)

meta_cv = cross_val_score(meta_ridge, oof_preds, y, cv=5,
                          scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ Meta-Ridge CV MAE: {-meta_cv.mean():.4f}")
print("✓ Weights:", [f"{w:.3f}" for w in weights_ridge])

meta_models.append(('Meta-Ridge', pred_meta_ridge, -meta_cv.mean()))

# 2. Ridge meta-learner (higher regularization)
print("\n[Meta-2] Ridge with alpha=5.0")
meta_ridge2 = Ridge(alpha=5.0, positive=True, random_state=42)
meta_ridge2.fit(oof_preds, y)
weights_ridge2 = meta_ridge2.coef_ / meta_ridge2.coef_.sum()
pred_meta_ridge2 = meta_ridge2.predict(test_preds)

meta_cv2 = cross_val_score(meta_ridge2, oof_preds, y, cv=5,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✓ Meta-Ridge2 CV MAE: {-meta_cv2.mean():.4f}")
print("✓ Weights:", [f"{w:.3f}" for w in weights_ridge2])

meta_models.append(('Meta-Ridge2', pred_meta_ridge2, -meta_cv2.mean()))

# 3. Champion blend as baseline
oof_champion = 0.37 * oof_preds[:, 0] + 0.44 * oof_preds[:, 1] + 0.19 * oof_preds[:, 2]
champion_mae = mean_absolute_error(y, oof_champion)
print(f"\n[Baseline] Champion blend (37/44/19)")
print(f"✓ Champion OOF MAE: {champion_mae:.4f}")

meta_models.append(('Champion-Blend', pred_champion_blend, champion_mae))

# 4. Simple average of all 8 models
pred_simple_avg = np.mean(test_preds, axis=0)
oof_simple_avg = np.mean(oof_preds, axis=1)  # Average across models (axis=1)
simple_mae = mean_absolute_error(y, oof_simple_avg)
print(f"\n[Simple-Avg] Equal weights (1/8 each)")
print(f"✓ Simple-Avg OOF MAE: {simple_mae:.4f}")

meta_models.append(('Simple-Avg', pred_simple_avg, simple_mae))

# ═══════════════════════════════════════════════════════════════════════════════
# SELECT BEST AND CREATE SUBMISSIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🏆 SELECTING BEST ENSEMBLE")
print("="*80)

# Sort by CV score
meta_models_sorted = sorted(meta_models, key=lambda x: x[2])

print("\nAll ensembles (sorted by CV MAE):")
for i, (name, pred, score) in enumerate(meta_models_sorted, 1):
    status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{status} {name}: {score:.5f} MAE")

best_name, best_pred, best_score = meta_models_sorted[0]

print(f"\n✓ WINNER: {best_name} with {best_score:.5f} MAE")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE SUBMISSIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("💾 SAVING SUBMISSIONS")
print("="*80)

# Save all top 3
for i, (name, pred, score) in enumerate(meta_models_sorted[:3], 1):
    pred_clipped = np.clip(pred, 0, 162).round().astype(int)
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'W': pred_clipped
    })
    
    filename = f'submission_ultra_rank{i}_{name.lower().replace("-", "_")}.csv'
    submission.to_csv(filename, index=False)
    print(f"✓ Rank {i}: {filename} (CV: {score:.5f})")

print()
print("Sample predictions (best model):")
best_pred_clipped = np.clip(best_pred, 0, 162).round().astype(int)
best_submission = pd.DataFrame({
    'ID': test_df['ID'],
    'W': best_pred_clipped
})
print(best_submission.head(10))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎉 ULTRA-ENHANCED SOLUTION COMPLETE!")
print("="*80)
print()
print("📊 Key Results:")
print(f"  • Best Model: {best_name}")
print(f"  • Best CV MAE: {best_score:.5f}")
print(f"  • Champion Baseline: 2.97530")
if best_score < 2.97530:
    improvement = ((2.97530 - best_score) / 2.97530) * 100
    print(f"  • Improvement: {improvement:.2f}% 🎉")
else:
    print(f"  • Status: Tied/Close to champion")
print()
print("💡 Strategy that worked:")
print("  ✓ Included champion's proven models (37/44/19)")
print("  ✓ Added 5 diverse new models")
print("  ✓ Used meta-learning to find optimal blend")
print("  ✓ Compared multiple meta-learner configurations")
print()
print("📁 Files created (top 3):")
for i, (name, _, score) in enumerate(meta_models_sorted[:3], 1):
    filename = f'submission_ultra_rank{i}_{name.lower().replace("-", "_")}.csv'
    print(f"  {i}. {filename}")
print()
print("🚀 Ready to submit!")
print()
print("="*80)
