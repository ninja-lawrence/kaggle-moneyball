"""
Adversarial Validation
======================

Train a classifier to distinguish between train and test sets.
Features that help distinguish them indicate distribution shift.
We can then:
1. Remove/adjust features with high distribution shift
2. Add features to account for the shift
3. Weight training samples by similarity to test set
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("=" * 80)
print("ADVERSARIAL VALIDATION ANALYSIS")
print("=" * 80)

def create_features(df):
    """Create features matching the 2.98 winning model"""
    
    # Basic stats
    df['R_per_G'] = df['R'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['H_per_G'] = df['H'] / df['G']
    df['HR_per_G'] = df['HR'] / df['G']
    df['BB_per_G'] = df['BB'] / df['G']
    
    # Run differential
    df['run_diff'] = df['R'] - df['RA']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['run_ratio'] = df['R'] / (df['RA'] + 1)
    df['run_diff_sqrt'] = np.sign(df['run_diff']) * np.sqrt(np.abs(df['run_diff']))
    
    # Pythagorean expectation
    exponents = [1.83, 1.87, 1.90, 1.93, 1.97, 2.00]
    for exp in exponents:
        df[f'pyth_wins_{int(exp*100)}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp)) * df['G']
    
    # Offensive stats
    df['TB'] = df['H'] + df['2B'] + 2*df['3B'] + 3*df['HR']
    df['SLG'] = df['TB'] / (df['AB'] + 1)
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
    df['OPS'] = df['OBP'] + df['SLG']
    df['BA'] = df['H'] / (df['AB'] + 1)
    df['ISO'] = df['SLG'] - df['BA']
    
    # Pitching stats
    df['IP'] = df['IPouts'] / 3
    df['WHIP'] = (df['HA'] + df['BBA']) / (df['IP'] + 1)
    df['K_per_9'] = (df['SOA'] * 9) / (df['IP'] + 1)
    df['BB_per_9'] = (df['BBA'] * 9) / (df['IP'] + 1)
    df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
    
    return df

print("\n🔧 Creating features...")
train = create_features(train)
test = create_features(test)

# Exclude target and ID columns
exclude_cols = ['ID', 'W', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
                'mlb_rpg']

# Include temporal features for adversarial validation (to see if they help distinguish)
feature_cols = [col for col in train.columns if col not in exclude_cols]

# Prepare data for adversarial validation
X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Create labels: 0 = train, 1 = test
X_train['is_test'] = 0
X_test['is_test'] = 1

# Combine
X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_combined = X_combined['is_test']
X_combined = X_combined.drop('is_test', axis=1)

print(f"\n📊 Features: {len(X_combined.columns)}")
print(f"📏 Total samples: {len(X_combined)}")
print(f"   Train: {len(X_train) - 1} ({100 * (len(X_train)-1) / len(X_combined):.1f}%)")
print(f"   Test:  {len(X_test) - 1} ({100 * (len(X_test)-1) / len(X_combined):.1f}%)")

# ============================================================================
# STEP 1: Train Adversarial Classifier
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: TRAINING ADVERSARIAL CLASSIFIER")
print("=" * 80)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Try Random Forest first
print("\n🌲 Random Forest Classifier:")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf, X_scaled, y_combined, cv=5, scoring='roc_auc')
rf_auc = rf_scores.mean()
print(f"   Cross-val AUC: {rf_auc:.4f} (±{rf_scores.std():.4f})")

# Train final model
rf.fit(X_scaled, y_combined)

# Try Gradient Boosting
print("\n🚀 Gradient Boosting Classifier:")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_scores = cross_val_score(gb, X_scaled, y_combined, cv=5, scoring='roc_auc')
gb_auc = gb_scores.mean()
print(f"   Cross-val AUC: {gb_auc:.4f} (±{gb_scores.std():.4f})")

# Train final model
gb.fit(X_scaled, y_combined)

# Interpret AUC
print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print(f"\nAUC = {rf_auc:.4f}")
if rf_auc < 0.55:
    print("✅ EXCELLENT: Train and test are very similar!")
    print("   Distribution shift is minimal.")
elif rf_auc < 0.65:
    print("✅ GOOD: Small distribution shift.")
    print("   Some differences but manageable.")
elif rf_auc < 0.75:
    print("⚠️  MODERATE: Noticeable distribution shift.")
    print("   Train and test differ significantly.")
else:
    print("🚨 HIGH: Major distribution shift!")
    print("   Train and test are quite different.")

# ============================================================================
# STEP 2: Feature Importance Analysis
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FEATURES WITH DISTRIBUTION SHIFT")
print("=" * 80)

# Get feature importance from both models
rf_importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance_rf': rf.feature_importances_
}).sort_values('importance_rf', ascending=False)

gb_importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance_gb': gb.feature_importances_
}).sort_values('importance_gb', ascending=False)

# Merge
importance = rf_importance.merge(gb_importance, on='feature')
importance['avg_importance'] = (importance['importance_rf'] + importance['importance_gb']) / 2
importance = importance.sort_values('avg_importance', ascending=False)

print("\nTop 20 features that distinguish train from test:")
print("(High importance = feature distributions differ between train/test)")
print("=" * 80)
for idx, row in importance.head(20).iterrows():
    print(f"{row['feature']:30s}: RF={row['importance_rf']:.4f}, GB={row['importance_gb']:.4f}, Avg={row['avg_importance']:.4f}")

# ============================================================================
# STEP 3: Analyze Distribution Shifts
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: STATISTICAL ANALYSIS OF TOP SHIFTED FEATURES")
print("=" * 80)

# Get back to original train/test
X_train_orig = X_train.drop('is_test', axis=1)
X_test_orig = X_test.drop('is_test', axis=1)

print("\nComparing train vs test statistics for problematic features:\n")
for feature in importance.head(10)['feature']:
    train_mean = X_train_orig[feature].mean()
    test_mean = X_test_orig[feature].mean()
    train_std = X_train_orig[feature].std()
    test_std = X_test_orig[feature].std()
    
    # Calculate percent difference
    if train_mean != 0:
        pct_diff = ((test_mean - train_mean) / abs(train_mean)) * 100
    else:
        pct_diff = 0
    
    print(f"\n{feature}:")
    print(f"  Train: mean={train_mean:.4f}, std={train_std:.4f}")
    print(f"  Test:  mean={test_mean:.4f}, std={test_std:.4f}")
    print(f"  Difference: {pct_diff:+.2f}%")

# ============================================================================
# STEP 4: Temporal Information (if available)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TEMPORAL INFORMATION")
print("=" * 80)

# Reload original data
train_orig = pd.read_csv('data/train.csv')
test_orig = pd.read_csv('data/test.csv')

print(f"\n🔍 KEY FINDING: Test set has NO yearID/temporal info!")
print(f"   This means the model must work across ALL eras.")
print(f"   Temporal features (decade/era) might hurt if test spans multiple eras.")

print(f"\nTrain years: {train_orig['yearID'].min()} - {train_orig['yearID'].max()}")
print("\nTrain decade distribution:")
print(train_orig['decade_label'].value_counts().sort_index())

print("\n💡 Insight: Since test has no temporal indicators, it likely")
print("   contains teams from various eras. This validates why removing")
print("   temporal features improved your 2.98 model!")

# ============================================================================
# STEP 5: Sample Weighting Strategy
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SAMPLE WEIGHTING BASED ON SIMILARITY TO TEST")
print("=" * 80)

# Predict probability of being in test set for training samples
train_probs = rf.predict_proba(scaler.transform(X_train_orig))[:, 1]

# Higher probability = more similar to test set = higher weight
train['similarity_weight'] = train_probs
train['similarity_weight_scaled'] = (train['similarity_weight'] - train['similarity_weight'].min()) / \
                                     (train['similarity_weight'].max() - train['similarity_weight'].min())

print(f"\nSimilarity weights (higher = more like test set):")
print(f"  Mean: {train['similarity_weight_scaled'].mean():.4f}")
print(f"  Std:  {train['similarity_weight_scaled'].std():.4f}")
print(f"  Min:  {train['similarity_weight_scaled'].min():.4f}")
print(f"  Max:  {train['similarity_weight_scaled'].max():.4f}")

# Show most/least similar teams
print("\n10 Training teams MOST similar to test set:")
most_similar = train.nlargest(10, 'similarity_weight')[['yearID', 'teamID', 'W', 
                                                          'similarity_weight_scaled', 'decade_label']]
print(most_similar.to_string(index=False))

print("\n10 Training teams LEAST similar to test set:")
least_similar = train.nsmallest(10, 'similarity_weight')[['yearID', 'teamID', 'W', 
                                                            'similarity_weight_scaled', 'decade_label']]
print(least_similar.to_string(index=False))

# ============================================================================
# STEP 6: Build Model with Adversarial Insights
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TRAINING MODEL WITH ADVERSARIAL WEIGHTING")
print("=" * 80)

# Remove features with high distribution shift
high_shift_features = importance.head(5)['feature'].tolist()
print(f"\n🗑️  Removing top 5 shifted features: {', '.join(high_shift_features)}")

feature_cols_filtered = [col for col in feature_cols if col not in high_shift_features]

X = train[feature_cols_filtered]
y = train['W']
X_test_final = test[feature_cols_filtered]

# Use similarity weights
weights = train['similarity_weight_scaled']

print(f"\n📊 Features: {len(feature_cols_filtered)} (removed {len(high_shift_features)})")
print(f"📏 Training samples: {len(X)}")

# Test with weighted model
from sklearn.model_selection import KFold

alphas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
seeds = [42, 123, 456]

best_score = float('inf')
best_alpha = None

for alpha in alphas:
    scores = []
    
    for seed in seeds:
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = weights.iloc[train_idx]
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_tr_scaled, y_tr, sample_weight=w_tr)
            
            pred = model.predict(X_val_scaled)
            mae = np.mean(np.abs(y_val - pred))
            fold_scores.append(mae)
        
        scores.append(np.mean(fold_scores))
    
    avg_score = np.mean(scores)
    
    if avg_score < best_score:
        best_score = avg_score
        best_alpha = alpha
    
    print(f"  Alpha {alpha:5.1f}: CV MAE = {avg_score:.4f} (±{np.std(scores):.4f})")

print(f"\n✅ Best alpha: {best_alpha}")
print(f"✅ Best CV MAE: {best_score:.4f}")

# Train final model
print(f"\n🏋️ Training final model with alpha={best_alpha}...")

all_predictions = []

for seed in seeds:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test_final)
    
    model = Ridge(alpha=best_alpha, random_state=seed)
    model.fit(X_scaled, y, sample_weight=weights)
    
    pred = model.predict(X_test_scaled)
    all_predictions.append(pred)
    print(f"  Seed {seed}: trained")

final_predictions = np.mean(all_predictions, axis=0)

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'ID': test['ID'],
    'W': final_predictions
})

submission.to_csv('submission_adversarial.csv', index=False)

print("\n" + "=" * 80)
print("✅ SUBMISSION CREATED: submission_adversarial.csv")
print("=" * 80)
print(f"📊 Features used: {len(feature_cols_filtered)}")
print(f"🗑️  Features removed: {len(high_shift_features)}")
print(f"⚖️  Adversarial weighting: Applied")
print(f"🎯 Best alpha: {best_alpha}")
print(f"📉 CV MAE: {best_score:.4f}")
print(f"🌱 Seeds used: {len(seeds)}")

print("\nKey improvements:")
print(f"  1. ✅ Removed {len(high_shift_features)} features with high distribution shift")
print(f"  2. ✅ Weighted training samples by similarity to test set")
print(f"  3. ✅ AUC = {rf_auc:.4f} indicates {'low' if rf_auc < 0.65 else 'moderate' if rf_auc < 0.75 else 'high'} shift")

print("\nPrediction statistics:")
print(f"  Mean: {final_predictions.mean():.2f}")
print(f"  Std:  {final_predictions.std():.2f}")
print(f"  Min:  {final_predictions.min():.2f}")
print(f"  Max:  {final_predictions.max():.2f}")

print("\n" + "=" * 80)
print("📊 Check the output above for distribution shift insights!")
print("=" * 80)
