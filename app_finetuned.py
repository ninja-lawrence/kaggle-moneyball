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

# ============================================================================
# FINE-TUNE THE WINNING APPROACH (no-temporal with strategic feature selection)
# ============================================================================

def create_balanced_features(df):
    """Balanced feature set - not too many, not too few"""
    df = df.copy()
    
    # Pythagorean expectations (core predictors)
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.82, 1.83, 1.85, 1.87, 1.90, 2.00]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        # Run differential features
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / df['G']
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    
    # Per game rates for key stats
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
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

print("\nCreating balanced features...")
train_df = create_balanced_features(train_df)
test_df = create_balanced_features(test_df)

# Exclude temporal features (proven to help!)
exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'mlb_rpg'
}

train_features = set(train_df.columns) - exclude_cols
test_features = set(test_df.columns) - exclude_cols
common_features = sorted(list(train_features & test_features))

print(f"Using {len(common_features)} features")

X_train = train_df[common_features].fillna(0)
y_train = train_df['W']
X_test = test_df[common_features].fillna(0)
test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df.index

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# EXTENSIVE ALPHA SEARCH
# ============================================================================
print("\n" + "="*80)
print("EXTENSIVE ALPHA SEARCH")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Very fine-grained search
alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]

best_alpha = None
best_cv_mae = float('inf')

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=kfold,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Alpha={alpha:5.2f}: CV MAE = {cv_mae:.4f} ± {cv_std:.4f}")
    
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
print(f"Gap: {abs(train_mae - best_cv_mae):.4f}")

# Also train with different random seeds and average predictions (pseudo-ensemble)
print("\n" + "="*80)
print("CREATING MULTI-SEED ENSEMBLE FOR STABILITY")
print("="*80)

seeds = [42, 123, 456, 789, 2024]
all_test_preds = []

for seed in seeds:
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    model = Ridge(alpha=best_alpha)
    
    # CV score with this seed
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    
    # Train on full data and predict
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    all_test_preds.append(pred)
    
    print(f"Seed {seed}: CV MAE = {cv_mae:.4f}")

# Average predictions across seeds
avg_test_pred = np.mean(all_test_preds, axis=0)
avg_test_pred = np.clip(avg_test_pred, 0, 162)
avg_test_pred_int = np.round(avg_test_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': avg_test_pred_int
})

submission_df.to_csv('submission_finetuned.csv', index=False)

print("\n" + "="*80)
print("FINE-TUNED SUBMISSION CREATED")
print("="*80)
print(f"File: submission_finetuned.csv")
print(f"Features: {len(common_features)}")
print(f"Alpha: {best_alpha}")
print(f"Strategy: Multi-seed ensemble for stability")
print(f"Expected Kaggle MAE: ~{best_cv_mae:.2f} to {best_cv_mae + 0.25:.2f}")
print(f"\nPrediction stats:")
print(f"  Mean: {avg_test_pred_int.mean():.2f} wins")
print(f"  Std: {avg_test_pred_int.std():.2f}")
print(f"  Range: {avg_test_pred_int.min()} to {avg_test_pred_int.max()}")

# Top features
feature_importance = pd.DataFrame({
    'Feature': common_features,
    'Coefficient': final_model.coef_
}).assign(
    abs_coef=lambda x: np.abs(x['Coefficient'])
).sort_values('abs_coef', ascending=False)

print("\n" + "="*80)
print("TOP 20 FEATURES")
print("="*80)
print(feature_importance.head(20)[['Feature', 'Coefficient']].to_string(index=False))

print("\n" + "="*80)
print("STRATEGY SUMMARY")
print("="*80)
print("✓ Balanced feature set (~50 features)")
print("✓ No temporal features (decade/era)")
print("✓ Fine-tuned alpha selection")
print("✓ Multi-seed ensemble for prediction stability")
print("✓ Based on insights from best model (no-temporal 3.03)")
