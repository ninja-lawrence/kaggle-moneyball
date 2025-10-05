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
# ENSEMBLE: COMBINE MULTIPLE FEATURE SETS
# ============================================================================

def create_feature_set_1(df):
    """Pythagorean-focused features"""
    df = df.copy()
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.82, 1.83, 1.85, 1.90, 2.00]:
            exp_str = str(int(exp * 100))
            df[f'pyth_wins_{exp_str}'] = (df['R']**exp / (df['R']**exp + df['RA']**exp + 1)) * df['G']
        df['run_diff_per_game'] = (df['R'] - df['RA']) / df['G']
    return df

def create_feature_set_2(df):
    """Volume and efficiency features"""
    df = df.copy()
    features = {}
    if 'G' in df.columns:
        for col in ['R', 'RA']:
            if col in df.columns:
                features[f'{col}_per_G'] = df[col] / df['G']
    if 'H' in df.columns and 'AB' in df.columns and 'BB' in df.columns:
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            features['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            features['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
    for k, v in features.items():
        df[k] = v
    return df

def clean_features(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df

print("\nCreating multiple feature sets...")
train_set1 = create_feature_set_1(train_df.copy())
train_set2 = create_feature_set_2(train_df.copy())
test_set1 = create_feature_set_1(test_df.copy())
test_set2 = create_feature_set_2(test_df.copy())

train_set1 = clean_features(train_set1)
train_set2 = clean_features(train_set2)
test_set1 = clean_features(test_set1)
test_set2 = clean_features(test_set2)

# Prepare feature sets
exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'mlb_rpg', 'H', 'AB', 'R', 'RA', 'HR', 'BB', '2B', '3B', 'SO',
    'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'SB', 'FP',
}

# Feature Set 1: Pythagorean focus
features1 = sorted(list((set(train_set1.columns) & set(test_set1.columns)) - exclude_cols))
# Feature Set 2: Volume/efficiency focus
features2 = sorted(list((set(train_set2.columns) & set(test_set2.columns)) - exclude_cols))

y_train = train_df['W']
test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df.index
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

print(f"Feature Set 1 (Pythagorean): {len(features1)} features")
print(f"Feature Set 2 (Volume/Efficiency): {len(features2)} features")

# ============================================================================
# TRAIN MODELS ON EACH FEATURE SET
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS ON DIFFERENT FEATURE SETS")
print("="*80)

# Model 1: Pythagorean features
X_train1 = train_set1[features1].fillna(0)
X_test1 = test_set1[features1].fillna(0)
scaler1 = StandardScaler()
X_train1_scaled = scaler1.fit_transform(X_train1)
X_test1_scaled = scaler1.transform(X_test1)

model1 = Ridge(alpha=3.0)
cv_scores1 = cross_val_score(model1, X_train1_scaled, y_train, cv=kfold,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae1 = -cv_scores1.mean()
print(f"Model 1 (Pythagorean focus): CV MAE = {cv_mae1:.4f}")

model1.fit(X_train1_scaled, y_train)
pred1_test = model1.predict(X_test1_scaled)

# Model 2: Volume/efficiency features
X_train2 = train_set2[features2].fillna(0)
X_test2 = test_set2[features2].fillna(0)
scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

model2 = Ridge(alpha=3.0)
cv_scores2 = cross_val_score(model2, X_train2_scaled, y_train, cv=kfold,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae2 = -cv_scores2.mean()
print(f"Model 2 (Volume/Efficiency): CV MAE = {cv_mae2:.4f}")

model2.fit(X_train2_scaled, y_train)
pred2_test = model2.predict(X_test2_scaled)

# ============================================================================
# FIND OPTIMAL ENSEMBLE WEIGHT
# ============================================================================
print("\n" + "="*80)
print("FINDING OPTIMAL ENSEMBLE WEIGHTS")
print("="*80)

best_weight1 = None
best_ensemble_cv = float('inf')

for weight1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    weight2 = 1 - weight1
    
    fold_maes = []
    for train_idx, val_idx in kfold.split(X_train1):
        # Train models on fold
        m1_fold = Ridge(alpha=3.0)
        m1_fold.fit(X_train1_scaled[train_idx], y_train.iloc[train_idx])
        p1_val = m1_fold.predict(X_train1_scaled[val_idx])
        
        m2_fold = Ridge(alpha=3.0)
        m2_fold.fit(X_train2_scaled[train_idx], y_train.iloc[train_idx])
        p2_val = m2_fold.predict(X_train2_scaled[val_idx])
        
        # Ensemble predictions
        ensemble_val = weight1 * p1_val + weight2 * p2_val
        mae = mean_absolute_error(y_train.iloc[val_idx], ensemble_val)
        fold_maes.append(mae)
    
    avg_mae = np.mean(fold_maes)
    print(f"Weight1={weight1:.1f}, Weight2={weight2:.1f}: CV MAE = {avg_mae:.4f}")
    
    if avg_mae < best_ensemble_cv:
        best_ensemble_cv = avg_mae
        best_weight1 = weight1

print(f"\nBest ensemble: Model1={best_weight1:.1f}, Model2={1-best_weight1:.1f}")
print(f"Ensemble CV MAE: {best_ensemble_cv:.4f}")
print(f"vs Model1 alone: {cv_mae1:.4f}")
print(f"vs Model2 alone: {cv_mae2:.4f}")

# Create final ensemble prediction
ensemble_pred = best_weight1 * pred1_test + (1 - best_weight1) * pred2_test
ensemble_pred = np.clip(ensemble_pred, 0, 162)
ensemble_pred_int = np.round(ensemble_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': ensemble_pred_int
})

submission_df.to_csv('submission_multi_ensemble.csv', index=False)

print("\n" + "="*80)
print("MULTI-MODEL ENSEMBLE SUBMISSION CREATED")
print("="*80)
print(f"File: submission_multi_ensemble.csv")
print(f"Expected Kaggle MAE: ~{best_ensemble_cv:.2f} to {best_ensemble_cv + 0.25:.2f}")
print(f"\nPrediction stats:")
print(f"  Mean: {ensemble_pred_int.mean():.2f} wins")
print(f"  Std: {ensemble_pred_int.std():.2f}")
print(f"  Range: {ensemble_pred_int.min()} to {ensemble_pred_int.max()}")
