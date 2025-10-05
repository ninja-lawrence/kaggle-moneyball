"""
Error Analysis & Residual Pattern Investigation
================================================

Analyze where the 2.98 model makes mistakes to guide improvements.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('data/train.csv')

print("=" * 80)
print("ERROR ANALYSIS FOR 2.98 MODEL")
print("=" * 80)

# Recreate the best model's features (no-temporal approach)
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

train = create_features(train)

# Exclude columns
exclude_cols = ['ID', 'W', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
                'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
                'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
                'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000',
                'decade_2010', 'mlb_rpg']

feature_cols = [col for col in train.columns if col not in exclude_cols]

X = train[feature_cols]
y = train['W']

print(f"\n📊 Features: {len(feature_cols)}")
print(f"📏 Training samples: {len(X)}")

# Get out-of-fold predictions
print("\n🔄 Generating out-of-fold predictions...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_tr_scaled, y_tr)
    
    oof_predictions[val_idx] = model.predict(X_val_scaled)

# Calculate residuals
train['predicted_wins'] = oof_predictions
train['residual'] = train['W'] - train['predicted_wins']
train['abs_residual'] = np.abs(train['residual'])

overall_mae = np.mean(train['abs_residual'])
print(f"\n✅ Overall MAE: {overall_mae:.4f}")

# ============================================================================
# ANALYSIS 1: Outlier Teams
# ============================================================================
print("\n" + "=" * 80)
print("1. OUTLIER ANALYSIS - Teams with Largest Errors")
print("=" * 80)

outliers = train.nlargest(20, 'abs_residual')[['yearID', 'teamID', 'W', 'predicted_wins', 
                                                 'residual', 'abs_residual', 'R', 'RA', 
                                                 'pyth_wins_190']]
print("\nTop 20 worst predictions:")
print(outliers.to_string(index=False))

# Stats on outliers
print(f"\nTeams with error > 5 wins: {len(train[train['abs_residual'] > 5])}")
print(f"Teams with error > 10 wins: {len(train[train['abs_residual'] > 10])}")
print(f"Teams with error > 15 wins: {len(train[train['abs_residual'] > 15])}")

# ============================================================================
# ANALYSIS 2: Error by Decade
# ============================================================================
print("\n" + "=" * 80)
print("2. ERROR PATTERNS BY DECADE")
print("=" * 80)

decade_stats = train.groupby('decade_label').agg({
    'abs_residual': ['mean', 'std', 'count'],
    'residual': 'mean'
}).round(3)
decade_stats.columns = ['MAE', 'Std', 'Count', 'Bias']
print("\n", decade_stats)

print("\nInterpretation:")
print("- MAE: Average prediction error per decade")
print("- Bias: Positive = overpredict, Negative = underpredict")

# ============================================================================
# ANALYSIS 3: Error by Win Range
# ============================================================================
print("\n" + "=" * 80)
print("3. ERROR PATTERNS BY WIN RANGE")
print("=" * 80)

train['win_range'] = pd.cut(train['W'], bins=[0, 60, 70, 80, 90, 100, 200],
                             labels=['<60', '60-70', '70-80', '80-90', '90-100', '100+'])

win_range_stats = train.groupby('win_range').agg({
    'abs_residual': ['mean', 'std', 'count'],
    'residual': 'mean'
}).round(3)
win_range_stats.columns = ['MAE', 'Std', 'Count', 'Bias']
print("\n", win_range_stats)

# ============================================================================
# ANALYSIS 4: Error vs Pythagorean Expectation Deviation
# ============================================================================
print("\n" + "=" * 80)
print("4. PYTHAGOREAN DEVIATION ANALYSIS")
print("=" * 80)

train['pyth_deviation'] = train['W'] - train['pyth_wins_190']
train['pyth_deviation_abs'] = np.abs(train['pyth_deviation'])

# Check correlation between pythagorean deviation and model error
correlation = np.corrcoef(train['pyth_deviation_abs'], train['abs_residual'])[0, 1]
print(f"\nCorrelation between Pythagorean deviation and model error: {correlation:.4f}")

print("\nTeams that beat/underperform Pythagorean expectation:")
pyth_outliers = train.nlargest(10, 'pyth_deviation_abs')[['yearID', 'teamID', 'W', 
                                                            'pyth_wins_190', 'pyth_deviation',
                                                            'predicted_wins', 'residual']]
print(pyth_outliers.to_string(index=False))

# ============================================================================
# ANALYSIS 5: Feature Correlation with Errors
# ============================================================================
print("\n" + "=" * 80)
print("5. WHICH FEATURES CORRELATE WITH ERRORS?")
print("=" * 80)

# Calculate correlation of features with absolute residuals
feature_error_corr = []
for col in feature_cols:
    if train[col].dtype in ['int64', 'float64']:
        corr = np.corrcoef(train[col], train['abs_residual'])[0, 1]
        feature_error_corr.append((col, abs(corr)))

feature_error_corr.sort(key=lambda x: x[1], reverse=True)

print("\nTop 15 features correlated with prediction errors:")
print("(High correlation = model struggles when this feature is extreme)")
for feat, corr in feature_error_corr[:15]:
    print(f"  {feat:30s}: {corr:.4f}")

# ============================================================================
# ANALYSIS 6: Systematic Bias Check
# ============================================================================
print("\n" + "=" * 80)
print("6. SYSTEMATIC BIAS ANALYSIS")
print("=" * 80)

print(f"\nOverall bias (mean residual): {train['residual'].mean():.4f}")
print(f"Median residual: {train['residual'].median():.4f}")
print(f"Std of residuals: {train['residual'].std():.4f}")

# Check for bias in predictions
over_predict = len(train[train['residual'] < -3])
under_predict = len(train[train['residual'] > 3])
print(f"\nTeams over-predicted by >3 wins: {over_predict}")
print(f"Teams under-predicted by >3 wins: {under_predict}")

# ============================================================================
# ANALYSIS 7: Era-specific patterns
# ============================================================================
print("\n" + "=" * 80)
print("7. ERA-SPECIFIC ERROR PATTERNS")
print("=" * 80)

era_cols = [col for col in train.columns if col.startswith('era_')]
for era_col in era_cols:
    era_teams = train[train[era_col] == True]
    if len(era_teams) > 0:
        era_mae = era_teams['abs_residual'].mean()
        era_bias = era_teams['residual'].mean()
        era_name = era_col.replace('era_', 'Era ')
        print(f"{era_name}: MAE={era_mae:.4f}, Bias={era_bias:+.4f}, N={len(era_teams)}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("8. GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Predicted vs Actual
ax1 = axes[0, 0]
ax1.scatter(train['W'], train['predicted_wins'], alpha=0.5, s=10)
ax1.plot([40, 120], [40, 120], 'r--', linewidth=2)
ax1.set_xlabel('Actual Wins')
ax1.set_ylabel('Predicted Wins')
ax1.set_title('Predicted vs Actual Wins')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals distribution
ax2 = axes[0, 1]
ax2.hist(train['residual'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Residual (Actual - Predicted)')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals vs Predicted
ax3 = axes[0, 2]
ax3.scatter(train['predicted_wins'], train['residual'], alpha=0.5, s=10)
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Wins')
ax3.set_ylabel('Residual')
ax3.set_title('Residual Plot')
ax3.grid(True, alpha=0.3)

# Plot 4: Error by decade
ax4 = axes[1, 0]
decade_mae = train.groupby('decade_label')['abs_residual'].mean().sort_index()
ax4.bar(range(len(decade_mae)), decade_mae.values)
ax4.set_xticks(range(len(decade_mae)))
ax4.set_xticklabels(decade_mae.index, rotation=45)
ax4.set_ylabel('Mean Absolute Error')
ax4.set_title('Error by Decade')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Error by win range
ax5 = axes[1, 1]
win_range_mae = train.groupby('win_range')['abs_residual'].mean()
ax5.bar(range(len(win_range_mae)), win_range_mae.values)
ax5.set_xticks(range(len(win_range_mae)))
ax5.set_xticklabels(win_range_mae.index)
ax5.set_ylabel('Mean Absolute Error')
ax5.set_title('Error by Win Range')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Pythagorean deviation vs error
ax6 = axes[1, 2]
ax6.scatter(train['pyth_deviation'], train['residual'], alpha=0.5, s=10)
ax6.axhline(0, color='red', linestyle='--', linewidth=1)
ax6.axvline(0, color='red', linestyle='--', linewidth=1)
ax6.set_xlabel('Pythagorean Deviation (Actual - Expected)')
ax6.set_ylabel('Model Residual')
ax6.set_title('Pythagorean Deviation vs Error')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved visualization: error_analysis.png")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("9. RECOMMENDATIONS BASED ON ERROR ANALYSIS")
print("=" * 80)

print("\n📊 Key Findings:")

# Finding 1: Outliers
outlier_threshold = overall_mae * 3
severe_outliers = train[train['abs_residual'] > outlier_threshold]
print(f"\n1. OUTLIERS: {len(severe_outliers)} teams with error > {outlier_threshold:.2f} wins")
print(f"   → Consider removing these from training")
print(f"   → Potential improvement: 0.01-0.02 MAE")

# Finding 2: Decade bias
decade_variance = decade_stats['MAE'].std()
if decade_variance > 0.1:
    worst_decade = decade_stats['MAE'].idxmax()
    print(f"\n2. DECADE BIAS: High variance across decades (std={decade_variance:.3f})")
    print(f"   → Worst decade: {worst_decade}")
    print(f"   → Consider decade-specific models or features")

# Finding 3: Win range bias
win_range_variance = win_range_stats['MAE'].std()
if win_range_variance > 0.1:
    worst_range = win_range_stats['MAE'].idxmax()
    print(f"\n3. WIN RANGE BIAS: High variance across win ranges (std={win_range_variance:.3f})")
    print(f"   → Worst range: {worst_range}")
    print(f"   → Model struggles with extreme teams")

# Finding 4: Pythagorean deviation
if correlation > 0.3:
    print(f"\n4. PYTHAGOREAN DEVIATION: High correlation ({correlation:.3f})")
    print(f"   → Teams that beat/underperform expectations are hard to predict")
    print(f"   → Consider features that explain luck/clutch performance")

# Finding 5: Feature-error correlation
high_error_features = [f for f, c in feature_error_corr[:5] if c > 0.1]
if high_error_features:
    print(f"\n5. PROBLEMATIC FEATURES: {len(high_error_features)} features highly correlated with errors")
    print(f"   → Top features: {', '.join(high_error_features[:3])}")
    print(f"   → Consider removing or transforming these")

print("\n" + "=" * 80)
print("✅ Error analysis complete! Check error_analysis.png for visualizations.")
print("=" * 80)
