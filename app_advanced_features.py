"""
Advanced Feature Engineering for Moneyball
==========================================

This script creates sophisticated baseball-specific features using:
1. Advanced sabermetric statistics
2. Efficiency and consistency metrics
3. Mathematical transformations (log, sqrt, reciprocal)
4. Ratio and interaction terms
5. Power features and polynomial terms
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("=" * 80)
print("ADVANCED FEATURE ENGINEERING APPROACH")
print("=" * 80)

def create_advanced_features(df):
    """Create comprehensive feature set with domain knowledge"""
    
    # ========================================================================
    # BASELINE: Core Statistics (normalized per game)
    # ========================================================================
    df['R_per_G'] = df['R'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['H_per_G'] = df['H'] / df['G']
    df['HR_per_G'] = df['HR'] / df['G']
    df['BB_per_G'] = df['BB'] / df['G']
    df['SO_per_G'] = df['SO'] / df['G']
    
    # ========================================================================
    # 1. PYTHAGOREAN EXPECTATION (Multiple Exponents)
    # ========================================================================
    # The most important predictor - test many exponents
    exponents = [1.80, 1.83, 1.85, 1.87, 1.90, 1.93, 1.95, 1.97, 2.00, 2.03, 2.05]
    for exp in exponents:
        df[f'pyth_wins_{int(exp*100)}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp)) * df['G']
    
    # ========================================================================
    # 2. RUN DIFFERENTIAL FEATURES
    # ========================================================================
    df['run_diff'] = df['R'] - df['RA']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['run_ratio'] = df['R'] / (df['RA'] + 1)  # +1 to avoid division by zero
    df['run_diff_sqrt'] = np.sign(df['run_diff']) * np.sqrt(np.abs(df['run_diff']))
    df['run_diff_squared'] = df['run_diff'] ** 2
    df['run_product'] = df['R'] * df['RA']
    
    # ========================================================================
    # 3. OFFENSIVE EFFICIENCY (Sabermetrics)
    # ========================================================================
    # Total Bases
    df['TB'] = df['H'] + df['2B'] + 2*df['3B'] + 3*df['HR']
    
    # Slugging Percentage (SLG)
    df['SLG'] = df['TB'] / (df['AB'] + 1)
    
    # On-Base Percentage (OBP) - estimate without HBP and SF
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
    
    # On-Base Plus Slugging (OPS) - most important offensive stat
    df['OPS'] = df['OBP'] + df['SLG']
    
    # Batting Average (BA)
    df['BA'] = df['H'] / (df['AB'] + 1)
    
    # Isolated Power (ISO) - measures raw power
    df['ISO'] = df['SLG'] - df['BA']
    
    # Secondary Average (measures contribution beyond batting average)
    df['SecA'] = (df['BB'] + df['TB'] - df['H'] + df['SB']) / (df['AB'] + 1)
    
    # Power Factor (measures extra-base hit ability)
    df['PowerFactor'] = (df['2B'] + 2*df['3B'] + 3*df['HR']) / (df['H'] + 1)
    
    # Walk Rate
    df['BB_rate'] = df['BB'] / (df['AB'] + df['BB'] + 1)
    
    # Strikeout Rate
    df['SO_rate'] = df['SO'] / (df['AB'] + 1)
    
    # Contact Rate (inverse of strikeout rate)
    df['contact_rate'] = 1 - df['SO_rate']
    
    # Speed Score (stolen base component)
    df['SB_rate'] = df['SB'] / df['G']
    
    # ========================================================================
    # 4. PITCHING/DEFENSE METRICS
    # ========================================================================
    # Innings Pitched (approx)
    df['IP'] = df['IPouts'] / 3
    df['IP_per_game'] = df['IP'] / df['G']
    
    # Walks and Hits per Innings Pitched (WHIP)
    df['WHIP'] = (df['HA'] + df['BBA']) / (df['IP'] + 1)
    
    # Strikeouts per 9 innings
    df['K_per_9'] = (df['SOA'] * 9) / (df['IP'] + 1)
    
    # Walks per 9 innings
    df['BB_per_9'] = (df['BBA'] * 9) / (df['IP'] + 1)
    
    # Strikeout to Walk Ratio
    df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
    
    # Home Runs per 9 innings (HR/9)
    df['HR_per_9'] = (df['HRA'] * 9) / (df['IP'] + 1)
    
    # Hits per 9 innings
    df['H_per_9'] = (df['HA'] * 9) / (df['IP'] + 1)
    
    # Defense Independent Pitching Stats (DIPS) approximation
    df['DIPS_factor'] = (df['BBA'] + df['HRA']) / (df['IP'] + 1)
    
    # Fielding Independent Pitching (FIP) approximation
    # FIP = ((13*HR)+(3*BB)-(2*K))/IP + constant
    df['FIP'] = ((13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / (df['IP'] + 1)) + 3.2
    
    # ========================================================================
    # 5. DEFENSIVE EFFICIENCY
    # ========================================================================
    # Fielding Percentage is already in data as 'FP'
    df['error_rate'] = df['E'] / df['G']
    df['DP_rate'] = df['DP'] / df['G']
    
    # Defensive Efficiency Rating (DER) approximation
    # DER = (BFP - H - BB - HBP - K) / (BFP - BB - HBP - K)
    # We approximate BFP as AB + BB + HBP (no HBP data, so estimate)
    df['BFP_approx'] = df['AB'] + df['BB'] + df['HA'] + df['BBA']
    df['DER'] = (df['BFP_approx'] - df['HA'] - df['BBA'] - df['SOA']) / (df['BFP_approx'] - df['BBA'] - df['SOA'] + 1)
    
    # ========================================================================
    # 6. VOLUME/WORKLOAD METRICS
    # ========================================================================
    df['CG_rate'] = df['CG'] / df['G']
    df['SHO_rate'] = df['SHO'] / df['G']
    df['SV_rate'] = df['SV'] / df['G']
    
    # Complete game ratio (proxy for starter quality/depth)
    df['CG_ratio'] = df['CG'] / (df['G'] + 1)
    
    # Bullpen usage (inverse of CG)
    df['bullpen_usage'] = 1 - df['CG_ratio']
    
    # ========================================================================
    # 7. BALANCE METRICS (Offense vs Defense)
    # ========================================================================
    # Team balance between offense and defense
    df['offense_defense_ratio'] = df['R'] / (df['RA'] + 1)
    df['offense_defense_product'] = df['R'] * df['RA']
    df['offense_defense_diff'] = df['R'] - df['RA']
    
    # Pythagorean residual (difference from expected wins)
    df['pyth_expected_190'] = (df['R']**1.90 / (df['R']**1.90 + df['RA']**1.90)) * df['G']
    
    # ========================================================================
    # 8. CONSISTENCY/VARIANCE PROXIES
    # ========================================================================
    # We don't have game-by-game data, but can create proxies
    
    # Offensive consistency (higher OBP = more consistent)
    df['offensive_consistency'] = df['OBP'] * df['OPS']
    
    # Power consistency (singles vs extra bases)
    df['singles'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['extra_base_hits'] = df['2B'] + df['3B'] + df['HR']
    df['XBH_rate'] = df['extra_base_hits'] / (df['H'] + 1)
    
    # Contact quality (TB per hit)
    df['quality_of_contact'] = df['TB'] / (df['H'] + 1)
    
    # ========================================================================
    # 9. MATHEMATICAL TRANSFORMATIONS
    # ========================================================================
    # Log transformations (for skewed distributions)
    df['log_R'] = np.log1p(df['R'])
    df['log_RA'] = np.log1p(df['RA'])
    df['log_HR'] = np.log1p(df['HR'])
    df['log_SO'] = np.log1p(df['SO'])
    
    # Square root transformations
    df['sqrt_R'] = np.sqrt(df['R'])
    df['sqrt_RA'] = np.sqrt(df['RA'])
    df['sqrt_HR'] = np.sqrt(df['HR'])
    
    # Reciprocal features (for ratios)
    df['inv_RA'] = 1 / (df['RA'] + 1)
    df['inv_ERA'] = 1 / (df['ERA'] + 1)
    
    # ========================================================================
    # 10. INTERACTION FEATURES (Key Combinations)
    # ========================================================================
    # Offense * Defense interactions
    df['OPS_x_WHIP'] = df['OPS'] * df['WHIP']
    df['OPS_x_ERA'] = df['OPS'] * df['ERA']
    df['BA_x_FIP'] = df['BA'] * df['FIP']
    
    # Power * Pitching
    df['HR_x_HRA'] = df['HR'] * df['HRA']
    df['ISO_x_WHIP'] = df['ISO'] * df['WHIP']
    
    # Efficiency combinations
    df['OBP_x_K_BB_ratio'] = df['OBP'] * df['K_BB_ratio']
    df['SLG_x_WHIP'] = df['SLG'] * df['WHIP']
    
    # ========================================================================
    # 11. POLYNOMIAL FEATURES (Squared Terms)
    # ========================================================================
    df['OPS_squared'] = df['OPS'] ** 2
    df['WHIP_squared'] = df['WHIP'] ** 2
    df['run_ratio_squared'] = df['run_ratio'] ** 2
    
    # ========================================================================
    # 12. CONTEXT-ADJUSTED METRICS
    # ========================================================================
    # Runs vs League Average (already in data as mlb_rpg, but we exclude it for overfitting)
    # Instead, create our own normalization
    
    # Offensive performance relative to defense faced
    df['offense_vs_defense'] = (df['R'] / df['G']) / (df['RA'] / df['G'] + 1)
    
    # Pitching performance relative to offense faced
    df['pitching_quality'] = 1 / (df['ERA'] + 1)
    
    return df

# ========================================================================
# APPLY FEATURE ENGINEERING
# ========================================================================
print("\n🔧 Creating advanced features...")
train = create_advanced_features(train)
test = create_advanced_features(test)

# ========================================================================
# FEATURE SELECTION
# ========================================================================
# Exclude ID, target, and temporal features
exclude_cols = ['ID', 'W', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
                'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
                'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
                'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000',
                'decade_2010', 'mlb_rpg']  # Exclude mlb_rpg - causes overfitting

# Get feature columns
feature_cols = [col for col in train.columns if col not in exclude_cols]

# Remove any infinite or NaN values
train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)

# Fill NaN with median
for col in feature_cols:
    median_val = train[col].median()
    train[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

X = train[feature_cols]
y = train['W']
X_test = test[feature_cols]

print(f"📊 Total features created: {len(feature_cols)}")
print(f"📏 Training samples: {len(X)}")

# ========================================================================
# MULTI-SEED ENSEMBLE WITH OPTIMAL RIDGE
# ========================================================================
print("\n🎯 Training multi-seed ensemble with advanced features...")

# Use multiple seeds for stability
seeds = [42, 123, 456, 789, 2024]

# Test multiple alphas
alphas = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]

best_score = float('inf')
best_alpha = None
best_model = None

for alpha in alphas:
    scores = []
    
    for seed in seeds:
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_tr_scaled, y_tr)
            
            # Predict
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

# ========================================================================
# FINAL TRAINING WITH BEST ALPHA
# ========================================================================
print(f"\n🏋️ Training final ensemble with alpha={best_alpha}...")

all_predictions = []

for seed in seeds:
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = Ridge(alpha=best_alpha, random_state=seed)
    model.fit(X_scaled, y)
    
    # Predict
    pred = model.predict(X_test_scaled)
    all_predictions.append(pred)
    
    print(f"  Seed {seed}: Model trained")

# Average predictions across all seeds
final_predictions = np.mean(all_predictions, axis=0)

# ========================================================================
# FEATURE IMPORTANCE ANALYSIS
# ========================================================================
print("\n📊 Top 20 Most Important Features:")
print("=" * 60)

# Get coefficients from last model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=best_alpha, random_state=42)
model.fit(X_scaled, y)

# Calculate absolute importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:30s}: {row['coefficient']:8.4f}")

# ========================================================================
# CREATE SUBMISSION
# ========================================================================
submission = pd.DataFrame({
    'ID': test['ID'],
    'W': final_predictions
})

submission.to_csv('submission_advanced_features.csv', index=False)

print("\n" + "=" * 80)
print("✅ SUBMISSION CREATED: submission_advanced_features.csv")
print("=" * 80)
print(f"📊 Features used: {len(feature_cols)}")
print(f"🎯 Best alpha: {best_alpha}")
print(f"📉 CV MAE: {best_score:.4f}")
print(f"🌱 Seeds used: {len(seeds)}")
print("\nPrediction statistics:")
print(f"  Mean: {final_predictions.mean():.2f}")
print(f"  Std:  {final_predictions.std():.2f}")
print(f"  Min:  {final_predictions.min():.2f}")
print(f"  Max:  {final_predictions.max():.2f}")
print("=" * 80)
