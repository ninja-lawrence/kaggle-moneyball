import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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
# STRATEGY: SIMPLIFY & FOCUS ON MOST GENERALIZABLE FEATURES
# ============================================================================

def create_core_features(df):
    """Create only the most stable features that generalize well"""
    df = df.copy()
    
    # Core Pythagorean features (proven to work)
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        # Multiple exponents for robustness
        for exp in [1.83, 2.0]:
            df[f'pyth_exp_{int(exp*100)}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{int(exp*100)}'] = df[f'pyth_exp_{int(exp*100)}'] * df['G']
        
        # Run differential - simple and effective
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / df['G']
    
    # Basic rates (only most important)
    if 'G' in df.columns:
        for col in ['R', 'RA']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / df['G']
    
    # Only include if they exist in BOTH train and test
    # Offensive efficiency
    if 'H' in df.columns and 'AB' in df.columns and 'BB' in df.columns:
        df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            df['OPS'] = df['OBP'] + df['SLG']
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

print("\nCreating features...")
train_df = create_core_features(train_df)
test_df = create_core_features(test_df)

# Get common features - exclude target and IDs
exclude_cols = {'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins'}
train_features = set(train_df.columns) - exclude_cols
test_features = set(test_df.columns) - exclude_cols
common_features = sorted(list(train_features & test_features))

print(f"Using {len(common_features)} features")
print(f"Features: {common_features[:10]}...")

# Prepare data
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
# TEST MULTIPLE MODELS WITH AGGRESSIVE REGULARIZATION
# ============================================================================
print("\n" + "="*80)
print("TESTING MULTIPLE REGULARIZATION APPROACHES")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

models_to_test = []

# Ridge with various alphas (including more aggressive)
for alpha in [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
    models_to_test.append(('Ridge', Ridge(alpha=alpha), alpha))

# Lasso for feature selection
for alpha in [0.1, 0.5, 1.0, 2.0]:
    models_to_test.append(('Lasso', Lasso(alpha=alpha, max_iter=10000), alpha))

# ElasticNet (combination of L1 and L2)
for alpha in [1.0, 3.0, 5.0, 10.0]:
    models_to_test.append(('ElasticNet', ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000), alpha))

best_model_name = None
best_model = None
best_alpha = None
best_cv_mae = float('inf')

for model_name, model, alpha in models_to_test:
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"{model_name:12} (α={alpha:5.1f}): CV MAE = {cv_mae:.4f} ± {cv_std:.4f}")
    
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_model_name = model_name
        best_model = model
        best_alpha = alpha

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name} with alpha={best_alpha}")
print(f"Best CV MAE: {best_cv_mae:.4f}")
print(f"{'='*80}")

# ============================================================================
# TRAIN FINAL MODEL
# ============================================================================
print("\nTraining final model...")
best_model.fit(X_train_scaled, y_train)

# Predictions
train_pred = best_model.predict(X_train_scaled)
test_pred = best_model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
print(f"Train MAE: {train_mae:.4f}")
print(f"CV MAE: {best_cv_mae:.4f}")
print(f"Difference: {train_mae - best_cv_mae:.4f} (smaller = less overfitting)")

# ============================================================================
# OPTIMIZED SUBMISSION
# ============================================================================
test_pred = np.clip(test_pred, 0, 162)
test_pred_int = np.round(test_pred).astype(int)

submission_df = pd.DataFrame({
    'ID': test_ids.astype(int),
    'W': test_pred_int
})

submission_df.to_csv('submission_optimized.csv', index=False)

print("\n" + "="*80)
print("OPTIMIZED SUBMISSION CREATED")
print("="*80)
print(f"File: submission_optimized.csv")
print(f"Model: {best_model_name} (alpha={best_alpha})")
print(f"Expected Kaggle MAE: ~{best_cv_mae:.2f} to {best_cv_mae + 0.3:.2f}")
print(f"\nFirst 5 rows:")
print(submission_df.head())
print(f"\nPrediction stats:")
print(f"  Mean: {test_pred_int.mean():.2f} wins")
print(f"  Std: {test_pred_int.std():.2f}")
print(f"  Range: {test_pred_int.min()} to {test_pred_int.max()}")

# Feature importance
if hasattr(best_model, 'coef_'):
    print("\n" + "="*80)
    print("TOP 15 FEATURES")
    print("="*80)
    feature_importance = pd.DataFrame({
        'Feature': common_features,
        'Coefficient': best_model.coef_
    }).assign(
        abs_coef=lambda x: np.abs(x['Coefficient'])
    ).sort_values('abs_coef', ascending=False)
    
    print(feature_importance.head(15)[['Feature', 'Coefficient']].to_string(index=False))

print("\n" + "="*80)
print("OPTIMIZATION STRATEGY")
print("="*80)
print("1. Reduced feature set to most stable predictors")
print("2. Tested aggressive regularization (higher alphas)")
print("3. Compared Ridge, Lasso, and ElasticNet")
print("4. Focus: minimize train/CV gap to improve generalization")
print(f"\nGoal: Close the ~0.33 gap between CV ({best_cv_mae:.2f}) and Kaggle")
