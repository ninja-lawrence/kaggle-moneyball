import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
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
# STRATEGY: NO TEMPORAL FEATURES (decade/era) - THEY MAY CAUSE OVERFITTING
# ============================================================================

def create_stable_features(df):
    """Create features but EXCLUDE temporal indicators"""
    df = df.copy()
    
    # Pythagorean expectation variations
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.83, 1.85, 1.9, 2.0]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        # Run differential
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / df['G']
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    
    # Rates per game
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

print("\nCreating features (excluding temporal indicators)...")
train_df = create_stable_features(train_df)
test_df = create_stable_features(test_df)

# EXPLICITLY EXCLUDE temporal features
exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    # Exclude decade indicators
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    # Exclude era indicators
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    # Exclude league context (might not generalize)
    'mlb_rpg'
}

train_features = set(train_df.columns) - exclude_cols
test_features = set(test_df.columns) - exclude_cols
common_features = sorted(list(train_features & test_features))

print(f"Using {len(common_features)} features (NO temporal/era features)")

X_train = train_df[common_features].fillna(0)
y_train = train_df['W']
X_test = test_df[common_features].fillna(0)
test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df.index

# Try both StandardScaler and RobustScaler
print("\n" + "="*80)
print("TESTING SCALERS AND ALPHAS")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

best_scaler_name = None
best_scaler = None
best_alpha = None
best_cv_mae = float('inf')

for scaler_name, scaler in [('Standard', StandardScaler()), ('Robust', RobustScaler())]:
    X_train_scaled = scaler.fit_transform(X_train)
    
    for alpha in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]:
        ridge = Ridge(alpha=alpha)
        cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=kfold,
                                     scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"{scaler_name:8} Scaler, α={alpha:5.1f}: CV MAE = {cv_mae:.4f} ± {cv_std:.4f}")
        
        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_scaler_name = scaler_name
            best_scaler = scaler
            best_alpha = alpha

print(f"\n{'='*80}")
print(f"BEST: {best_scaler_name} Scaler with alpha={best_alpha}")
print(f"Best CV MAE: {best_cv_mae:.4f}")
print(f"{'='*80}")

# Scale with best scaler
X_train_scaled = best_scaler.fit_transform(X_train)
X_test_scaled = best_scaler.transform(X_test)

# Train final model
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

train_pred = final_model.predict(X_train_scaled)
test_pred = final_model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
print(f"\nTrain MAE: {train_mae:.4f}")
print(f"CV MAE: {best_cv_mae:.4f}")
print(f"Generalization gap: {train_mae - best_cv_mae:.4f}")

# Create submission
test_pred = np.clip(test_pred, 0, 162)
test_pred_int = np.round(test_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': test_pred_int
})

submission_df.to_csv('submission_notemporal.csv', index=False)

print("\n" + "="*80)
print("NO-TEMPORAL SUBMISSION CREATED")
print("="*80)
print(f"File: submission_notemporal.csv")
print(f"Scaler: {best_scaler_name}")
print(f"Alpha: {best_alpha}")
print(f"Expected Kaggle MAE: ~{best_cv_mae:.2f} to {best_cv_mae + 0.3:.2f}")
print(f"\nFirst 5 rows:")
print(submission_df.head())
print(f"\nPrediction stats:")
print(f"  Mean: {test_pred_int.mean():.2f} wins")
print(f"  Std: {test_pred_int.std():.2f}")
print(f"  Range: {test_pred_int.min()} to {test_pred_int.max()}")

# Top features
feature_importance = pd.DataFrame({
    'Feature': common_features,
    'Coefficient': final_model.coef_
}).assign(
    abs_coef=lambda x: np.abs(x['Coefficient'])
).sort_values('abs_coef', ascending=False)

print("\n" + "="*80)
print("TOP 15 FEATURES")
print("="*80)
print(feature_importance.head(15)[['Feature', 'Coefficient']].to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("Removed decade/era features that might cause overfitting")
print("Focus on universal baseball metrics that work across all time periods")
print("This should improve generalization to test set")
