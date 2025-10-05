import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================================
# ULTRA-CLEAN: Only most predictive, stable features
# ============================================================================

def create_ultra_clean_features(df):
    """Only the most proven features, nothing fancy"""
    df = df.copy()
    
    # Pythagorean wins with fine-tuned exponents
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        # Test exponents between 1.8 and 2.0 in small increments
        for exp in [1.80, 1.82, 1.83, 1.85, 1.87, 1.90, 2.00]:
            exp_str = str(int(exp * 100))
            df[f'pyth_wins_{exp_str}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp + 1)) * df['G']
        
        # Run differential (simple and stable)
        df['run_diff_per_game'] = (df['R'] - df['RA']) / df['G']
    
    # Only rates per game for R and RA (most important)
    if 'G' in df.columns:
        if 'R' in df.columns:
            df['R_per_G'] = df['R'] / df['G']
        if 'RA' in df.columns:
            df['RA_per_G'] = df['RA'] / df['G']
    
    # Core offensive metrics (only if they exist in both train and test)
    if 'H' in df.columns and 'AB' in df.columns and 'BB' in df.columns:
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
            df['OPS'] = df['OBP'] + df['SLG']
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

print("\nCreating ultra-clean features...")
train_df = create_ultra_clean_features(train_df)
test_df = create_ultra_clean_features(test_df)

# Exclude temporal, IDs, and original stats that are captured in derived features
exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'mlb_rpg',
    # Exclude raw counting stats since we have rates
    'H', 'AB', 'R', 'RA', 'HR', 'BB', '2B', '3B', 'SO',
    'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'SB', 'FP',
    # Keep only core volume metrics
}

train_features = set(train_df.columns) - exclude_cols
test_features = set(test_df.columns) - exclude_cols
common_features = sorted(list(train_features & test_features))

print(f"Using {len(common_features)} ultra-clean features")
print(f"Features: {common_features}")

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
# FIND OPTIMAL ALPHA WITH FINE-GRAINED SEARCH
# ============================================================================
print("\n" + "="*80)
print("FINE-GRAINED ALPHA SEARCH")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Test alphas in finer increments around the optimal range
alphas = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

best_alpha = None
best_cv_mae = float('inf')
cv_results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=kfold,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    cv_results.append((alpha, cv_mae, cv_std))
    print(f"Alpha={alpha:6.2f}: CV MAE = {cv_mae:.4f} ± {cv_std:.4f}")
    
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_alpha = alpha

print(f"\n{'='*80}")
print(f"BEST: Alpha={best_alpha}, CV MAE={best_cv_mae:.4f}")
print(f"{'='*80}")

# Train final model
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

train_pred = final_model.predict(X_train_scaled)
test_pred = final_model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
print(f"\nTrain MAE: {train_mae:.4f}")
print(f"CV MAE: {best_cv_mae:.4f}")
print(f"Gap: {abs(train_mae - best_cv_mae):.4f} (lower is better)")

# Create submission
test_pred = np.clip(test_pred, 0, 162)
test_pred_int = np.round(test_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': test_pred_int
})

submission_df.to_csv('submission_ultraclean.csv', index=False)

print("\n" + "="*80)
print("ULTRA-CLEAN SUBMISSION CREATED")
print("="*80)
print(f"File: submission_ultraclean.csv")
print(f"Features: {len(common_features)} (minimal, high-quality set)")
print(f"Alpha: {best_alpha}")
print(f"Expected Kaggle MAE: ~{best_cv_mae:.2f} to {best_cv_mae + 0.25:.2f}")
print(f"\nFirst 5 rows:")
print(submission_df.head())
print(f"\nPrediction stats:")
print(f"  Mean: {test_pred_int.mean():.2f} wins")
print(f"  Std: {test_pred_int.std():.2f}")
print(f"  Range: {test_pred_int.min()} to {test_pred_int.max()}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': common_features,
    'Coefficient': final_model.coef_
}).assign(
    abs_coef=lambda x: np.abs(x['Coefficient'])
).sort_values('abs_coef', ascending=False)

print("\n" + "="*80)
print(f"ALL {len(common_features)} FEATURES (by importance)")
print("="*80)
print(feature_importance[['Feature', 'Coefficient']].to_string(index=False))

print("\n" + "="*80)
print("STRATEGY")
print("="*80)
print("✓ Removed temporal features (decade/era)")
print("✓ Removed raw counting stats (kept only rates and pythagorean)")
print("✓ Fine-tuned pythagorean exponents (1.80 to 2.00)")
print("✓ Kept only core volume metrics (CG, SHO, SV, IPouts, ERA, G)")
print("✓ Focused on features that work across all baseball eras")
print(f"\nGoal: Beat 3.03 by maximizing generalization")
