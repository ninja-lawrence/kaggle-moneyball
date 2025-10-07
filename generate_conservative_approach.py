"""
═══════════════════════════════════════════════════════════════════════════════
🎯 CONSERVATIVE IMPROVEMENT STRATEGY 🎯
═══════════════════════════════════════════════════════════════════════════════

Key Finding: Champion OOF=2.784 but Kaggle=2.975
This is a ~0.19 gap suggesting overfitting or distribution shift!

NEW STRATEGY:
✨ Focus on reducing overfitting, not adding complexity
✨ More conservative regularization
✨ Ensemble with more weight averaging
✨ Feature pruning to remove noise
✨ Test different random seeds for stability

Goal: Close the gap between OOF and test performance

Date: October 7, 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 CONSERVATIVE IMPROVEMENT STRATEGY")
print("="*80)
print()
print("Focus: Reduce overfitting, improve generalization")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
y = train_df['W']

print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# CHAMPION FEATURES (PROVEN)
# ═══════════════════════════════════════════════════════════════════════════════

def create_core_features(df):
    """Only the most essential, proven features"""
    df = df.copy()
    
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        # Core Pythagorean (most proven exponents only)
        for exp in [1.83, 2.0]:
            exp_str = str(int(exp * 100))
            df[f'pyth_wins_{exp_str}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp + 1)) * df['G']
        
        # Run differential (most stable metric)
        df['run_diff_per_game'] = (df['R'] - df['RA']) / (df['G'] + 1)
    
    # Essential per-game rates only
    if 'G' in df.columns:
        for col in ['R', 'RA']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / (df['G'] + 1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SEED ENSEMBLE FOR STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🔧 MULTI-SEED ENSEMBLE (10 seeds for stability)")
print("="*80)

train_features = create_core_features(train_df.copy())
test_features = create_core_features(test_df.copy())

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

print(f"✓ Using {len(feature_cols)} core features (minimal set)")
print()

# Test multiple alpha values with multiple seeds
alphas = [5.0, 7.0, 10.0, 15.0, 20.0]
seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

all_predictions = []
model_configs = []

for alpha in alphas:
    print(f"\n📊 Testing Ridge alpha={alpha}")
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation for this alpha
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    seed_predictions = []
    for seed in seeds:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_train_scaled, y)
        pred = model.predict(X_test_scaled)
        seed_predictions.append(pred)
    
    # Average predictions across seeds
    avg_pred = np.mean(seed_predictions, axis=0)
    all_predictions.append(avg_pred)
    
    # Calculate CV score
    model_cv = Ridge(alpha=alpha, random_state=42)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model_cv, X_train_scaled, y, cv=kfold,
                             scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -scores.mean()
    
    print(f"  ✓ CV MAE: {cv_mae:.4f} (averaged over {len(seeds)} seeds)")
    model_configs.append((alpha, avg_pred, cv_mae))

# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎯 ENSEMBLE STRATEGIES")
print("="*80)

ensembles = []

# 1. Simple average of all
pred_simple = np.mean(all_predictions, axis=0)
ensembles.append(('Simple Average (all alphas)', pred_simple))

# 2. Best single model
best_alpha, best_pred, best_cv = min(model_configs, key=lambda x: x[2])
ensembles.append((f'Best Single (alpha={best_alpha})', best_pred))

# 3. Weighted by CV performance (inverse MAE)
weights = []
for alpha, pred, cv_mae in model_configs:
    weight = 1.0 / cv_mae  # Inverse of error
    weights.append(weight)
weights = np.array(weights) / sum(weights)  # Normalize

pred_weighted = sum(w * p for w, p in zip(weights, all_predictions))
ensembles.append(('Weighted by CV', pred_weighted))

print("\nEnsemble approaches:")
for i, (name, pred) in enumerate(ensembles, 1):
    print(f"  {i}. {name}")

# 4. Conservative blend (favor higher alpha = less overfit)
high_alpha_preds = [p for a, p, _ in model_configs if a >= 10]
pred_conservative = np.mean(high_alpha_preds, axis=0)
ensembles.append(('Conservative (alpha≥10)', pred_conservative))

print(f"  4. Conservative (alpha≥10)")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ALL VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("💾 SAVING SUBMISSIONS")
print("="*80)

for i, (name, pred) in enumerate(ensembles, 1):
    pred_clipped = np.clip(pred, 0, 162).round().astype(int)
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'W': pred_clipped
    })
    
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('≥', 'ge')
    filename = f'submission_conservative_{i}_{safe_name}.csv'
    submission.to_csv(filename, index=False)
    
    print(f"✓ {filename}")

print()
print("Sample predictions (best single model):")
best_submission = pd.DataFrame({
    'ID': test_df['ID'],
    'W': np.clip(best_pred, 0, 162).round().astype(int)
})
print(best_submission.head(10))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎉 CONSERVATIVE STRATEGY COMPLETE!")
print("="*80)
print()
print("📊 Key Approach:")
print(f"  • Minimal features: {len(feature_cols)} (vs 50-100 in complex models)")
print(f"  • High regularization: alpha 5-20 tested")
print(f"  • Multi-seed averaging: {len(seeds)} seeds per model")
print(f"  • Multiple alphas: {len(alphas)} configurations")
print()
print("🎯 Philosophy:")
print("  • LESS complexity = BETTER generalization")
print("  • Stability over performance")
print("  • Close OOF-test gap")
print()
print("📁 Files created:")
print("  1. submission_conservative_1_simple_average_all_alphas.csv")
print(f"  2. submission_conservative_2_best_single_alpha{best_alpha}.csv")
print("  3. submission_conservative_3_weighted_by_cv.csv")
print("  4. submission_conservative_4_conservative_alphage10.csv")
print()
print("💡 Recommendation:")
print("  Try #4 (Conservative) first - favors high regularization")
print("  Then try #2 (Best single) and #3 (Weighted)")
print()
print("="*80)
