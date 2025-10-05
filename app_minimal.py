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
# ULTRA-MINIMAL: ONLY PYTHAGOREAN EXPECTATION
# ============================================================================

def create_minimal_features(df):
    """Only the most fundamental features"""
    df = df.copy()
    
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        # Just a few proven pythagorean variations
        df['pyth_wins_183'] = (df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83 + 1)) * df['G']
        df['pyth_wins_200'] = (df['R']**2.00 / (df['R']**2.00 + df['RA']**2.00 + 1)) * df['G']
        df['run_diff_per_game'] = (df['R'] - df['RA']) / df['G']
    
    return df

print("\nCreating minimal features...")
train_df = create_minimal_features(train_df)
test_df = create_minimal_features(test_df)

# Use ONLY the new features we created
feature_cols = ['pyth_wins_183', 'pyth_wins_200', 'run_diff_per_game']

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['W']
X_test = test_df[feature_cols].fillna(0)
test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df.index

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nUsing only {len(feature_cols)} features: {feature_cols}")
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")

# ============================================================================
# RIDGE WITH MINIMAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("ULTRA-MINIMAL MODEL: 3 PYTHAGOREAN FEATURES ONLY")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

best_alpha = None
best_cv_mae = float('inf')

for alpha in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=kfold,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Alpha={alpha:5.1f}: CV MAE = {cv_mae:.4f} ± {cv_std:.4f}")
    
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha}")
print(f"Best CV MAE: {best_cv_mae:.4f}")

# Train final model
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

# Predictions
train_pred = final_model.predict(X_train_scaled)
test_pred = final_model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
print(f"\nTrain MAE: {train_mae:.4f}")
print(f"CV MAE: {best_cv_mae:.4f}")
print(f"Overfitting gap: {train_mae - best_cv_mae:.4f}")

# Create submission
test_pred = np.clip(test_pred, 0, 162)
test_pred_int = np.round(test_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': test_pred_int
})

submission_df.to_csv('submission_minimal.csv', index=False)

print("\n" + "="*80)
print("MINIMAL SUBMISSION CREATED")
print("="*80)
print(f"File: submission_minimal.csv")
print(f"Features used: {len(feature_cols)} (vs 55+ in other models)")
print(f"Expected Kaggle MAE: ~{best_cv_mae:.2f} to {best_cv_mae + 0.3:.2f}")
print(f"\nCoefficients:")
for feat, coef in zip(feature_cols, final_model.coef_):
    print(f"  {feat:20s}: {coef:8.4f}")
print(f"\nPrediction stats:")
print(f"  Mean: {test_pred_int.mean():.2f} wins")
print(f"  Std: {test_pred_int.std():.2f}")
print(f"  Range: {test_pred_int.min()} to {test_pred_int.max()}")

print("\n" + "="*80)
print("PHILOSOPHY: OCCAM'S RAZOR")
print("="*80)
print("Sometimes the simplest model generalizes best!")
print("Pythagorean expectation alone captures ~95% of win variance")
print("Fewer features = less chance of overfitting to training quirks")
