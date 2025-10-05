import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED STACKING APPROACH - TARGETING 2.6")
print("="*80)

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def create_advanced_features(df):
    """More sophisticated feature engineering"""
    df = df.copy()
    
    # Pythagorean expectations with MORE exponent variations
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.78, 1.80, 1.82, 1.83, 1.84, 1.85, 1.87, 1.90, 1.92, 1.95, 2.00]:
            exp_str = str(int(exp * 100))
            pyth_exp = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_exp_{exp_str}'] = pyth_exp
            df[f'pyth_wins_{exp_str}'] = pyth_exp * df['G']
        
        # Run differential features
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / df['G']
        df['run_diff_sq'] = df['run_diff']**2
        df['run_diff_sqrt'] = np.sign(df['run_diff']) * np.sqrt(np.abs(df['run_diff']))
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
        df['run_product'] = df['R'] * df['RA'] / (df['G'] + 1)
    
    # Per game rates
    if 'G' in df.columns:
        for col in ['R', 'RA', 'H', 'HR', 'BB', 'SO', '2B', '3B']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / df['G']
    
    # Offensive power metrics
    if 'H' in df.columns and 'AB' in df.columns:
        df['BA'] = df['H'] / (df['AB'] + 1)
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            df['ISO'] = df['SLG'] - df['BA']  # Isolated power
            if 'OBP' in df.columns:
                df['OPS'] = df['OBP'] + df['SLG']
    
    # Pitching metrics
    if 'ERA' in df.columns and 'IPouts' in df.columns:
        if 'HA' in df.columns and 'BBA' in df.columns:
            df['WHIP'] = (df['HA'] + df['BBA']) / ((df['IPouts'] / 3) + 1)
        if 'SOA' in df.columns:
            df['K_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 1)
            if 'BBA' in df.columns:
                df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
    
    # Interaction features
    if 'OPS' in df.columns and 'ERA' in df.columns:
        df['OPS_ERA_interaction'] = df['OPS'] / (df['ERA'] + 1)
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

print("\nCreating advanced features...")
train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

# Exclude temporal and ID columns
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
test_ids = test_df['ID']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STACKING ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("TRAINING BASE MODELS FOR STACKING")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define base models with different characteristics
base_models = [
    ('Ridge_0.5', Ridge(alpha=0.5)),
    ('Ridge_1', Ridge(alpha=1.0)),
    ('Ridge_3', Ridge(alpha=3.0)),
    ('Ridge_10', Ridge(alpha=10.0)),
    ('Lasso', Lasso(alpha=0.5, max_iter=10000)),
    ('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)),
    ('Huber', HuberRegressor(epsilon=1.1, alpha=0.5)),
    ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
]

# Generate out-of-fold predictions for stacking
oof_predictions = np.zeros((len(X_train), len(base_models)))
test_predictions = np.zeros((len(X_test), len(base_models)))

print(f"\nTraining {len(base_models)} base models...")

for i, (name, model) in enumerate(base_models):
    print(f"\n{i+1}. {name}")
    
    # Out-of-fold predictions
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        oof_predictions[val_idx, i] = model_clone.predict(X_val)
    
    # Train on full data for test predictions
    model.fit(X_train_scaled, y_train)
    test_predictions[:, i] = model.predict(X_test_scaled)
    
    # Evaluate
    oof_mae = mean_absolute_error(y_train, oof_predictions[:, i])
    print(f"   OOF MAE: {oof_mae:.4f}")

# ============================================================================
# META-MODEL (STACKING LAYER)
# ============================================================================
print("\n" + "="*80)
print("TRAINING META-MODEL")
print("="*80)

# Try different meta-models
meta_models = [
    ('Ridge', Ridge(alpha=1.0)),
    ('Lasso', Lasso(alpha=0.1, max_iter=10000)),
    ('ElasticNet', ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000)),
]

best_meta_name = None
best_meta_model = None
best_meta_mae = float('inf')

for name, meta_model in meta_models:
    cv_scores = []
    for train_idx, val_idx in kfold.split(oof_predictions):
        meta_train, meta_val = oof_predictions[train_idx], oof_predictions[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        meta_model_clone = type(meta_model)(**meta_model.get_params())
        meta_model_clone.fit(meta_train, y_tr)
        pred = meta_model_clone.predict(meta_val)
        cv_scores.append(mean_absolute_error(y_val, pred))
    
    avg_mae = np.mean(cv_scores)
    print(f"{name:12s}: CV MAE = {avg_mae:.4f}")
    
    if avg_mae < best_meta_mae:
        best_meta_mae = avg_mae
        best_meta_name = name
        best_meta_model = meta_model

print(f"\nBest meta-model: {best_meta_name}")
print(f"Stacked CV MAE: {best_meta_mae:.4f}")

# Train final meta-model
best_meta_model.fit(oof_predictions, y_train)
final_predictions = best_meta_model.predict(test_predictions)

# Clip and round
final_predictions = np.clip(final_predictions, 0, 162)
final_predictions_int = np.round(final_predictions).astype(int)

# Create submission
submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': final_predictions_int
})

submission_df.to_csv('submission_stacked.csv', index=False)

print("\n" + "="*80)
print("STACKED SUBMISSION CREATED")
print("="*80)
print(f"File: submission_stacked.csv")
print(f"Stacked CV MAE: {best_meta_mae:.4f}")
print(f"Expected Kaggle: ~{best_meta_mae:.2f} to {best_meta_mae + 0.25:.2f}")
print(f"\nPrediction stats:")
print(f"  Mean: {final_predictions_int.mean():.2f}")
print(f"  Std: {final_predictions_int.std():.2f}")
print(f"  Range: {final_predictions_int.min()}-{final_predictions_int.max()}")

print("\n" + "="*80)
print("STRATEGY")
print("="*80)
print("✓ Expanded pythagorean exponents (11 variations)")
print("✓ Added interaction features")
print("✓ Stacking ensemble with 9 diverse base models")
print("✓ Meta-model learns optimal combination")
print("✓ This approach can capture more complex patterns")
print("\nGoal: Push closer to leaderboard top (2.6)")
