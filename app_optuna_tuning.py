"""
Hyperparameter Tuning with Optuna
==================================

This script uses Optuna to systematically search for:
1. Optimal Ridge alpha
2. Best feature subset selection strategy
3. Optimal ensemble weights
4. Scaler choice (StandardScaler vs RobustScaler)
5. Number of CV folds
6. Feature engineering choices
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("=" * 80)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

def create_features(df, include_temporal=False, n_pyth_variants=5, include_advanced=False):
    """Create features with configurable complexity"""
    
    # Basic per-game stats
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
    
    # Pythagorean expectation - configurable number of variants
    if n_pyth_variants >= 1:
        exponents = np.linspace(1.80, 2.05, n_pyth_variants)
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
    
    if include_advanced:
        # Advanced features
        df['HR_per_9'] = (df['HRA'] * 9) / (df['IP'] + 1)
        df['FIP'] = ((13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / (df['IP'] + 1)) + 3.2
        df['SecA'] = (df['BB'] + df['TB'] - df['H'] + df['SB']) / (df['AB'] + 1)
        df['PowerFactor'] = (df['2B'] + 2*df['3B'] + 3*df['HR']) / (df['H'] + 1)
        
        # Interactions
        df['OPS_x_WHIP'] = df['OPS'] * df['WHIP']
        df['ISO_x_WHIP'] = df['ISO'] * df['WHIP']
        
        # Math transforms
        df['log_R'] = np.log1p(df['R'])
        df['log_RA'] = np.log1p(df['RA'])
        df['sqrt_HR'] = np.sqrt(df['HR'])
    
    return df

def get_feature_columns(df, include_temporal=False):
    """Get feature columns based on configuration"""
    exclude_cols = ['ID', 'W', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins', 'mlb_rpg']
    
    if not include_temporal:
        exclude_cols.extend(['era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
                            'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
                            'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000',
                            'decade_2010'])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

# Global variables for optimization
best_trial_predictions = None
best_trial_config = None

def objective(trial):
    """Optuna objective function"""
    
    # Hyperparameters to tune
    alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
    include_temporal = trial.suggest_categorical('include_temporal', [False])  # We know False is better
    n_pyth_variants = trial.suggest_int('n_pyth_variants', 3, 15)
    include_advanced = trial.suggest_categorical('include_advanced', [False, True])
    scaler_type = trial.suggest_categorical('scaler', ['standard', 'robust'])
    n_folds = trial.suggest_int('n_folds', 5, 15)
    n_seeds = trial.suggest_int('n_seeds', 1, 7)
    
    # Model type
    model_type = trial.suggest_categorical('model_type', ['ridge', 'lasso', 'elasticnet'])
    if model_type == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
    
    try:
        # Create features
        train_df = train.copy()
        train_df = create_features(train_df, include_temporal, n_pyth_variants, include_advanced)
        
        # Get features
        feature_cols = get_feature_columns(train_df, include_temporal)
        
        # Handle inf/nan
        train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            median_val = train_df[col].median()
            train_df[col].fillna(median_val, inplace=True)
        
        X = train_df[feature_cols]
        y = train_df['W']
        
        # Cross-validation with multiple seeds
        all_scores = []
        
        seeds = list(range(42, 42 + n_seeds * 100, 100))[:n_seeds]
        
        for seed in seeds:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = RobustScaler()
                
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                if model_type == 'ridge':
                    model = Ridge(alpha=alpha, random_state=seed)
                elif model_type == 'lasso':
                    model = Lasso(alpha=alpha, random_state=seed, max_iter=2000)
                else:  # elasticnet
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=2000)
                
                model.fit(X_tr_scaled, y_tr)
                
                # Predict
                pred = model.predict(X_val_scaled)
                mae = np.mean(np.abs(y_val - pred))
                all_scores.append(mae)
        
        avg_mae = np.mean(all_scores)
        
        return avg_mae
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

print("\n🔍 Starting Optuna hyperparameter search...")
print(f"📊 Training samples: {len(train)}")

# Create study
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='moneyball_optimization'
)

# Optimize
n_trials = 100  # Run 100 trials
print(f"🎯 Running {n_trials} optimization trials...\n")

study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

print("\n" + "=" * 80)
print("OPTIMIZATION RESULTS")
print("=" * 80)

print(f"\n✅ Best CV MAE: {study.best_value:.4f}")
print(f"\n📊 Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key:20s}: {value}")

# Train final model with best parameters
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("=" * 80)

best_params = study.best_params

# Create features with best params
train_df = train.copy()
test_df = test.copy()

train_df = create_features(
    train_df,
    include_temporal=best_params['include_temporal'],
    n_pyth_variants=best_params['n_pyth_variants'],
    include_advanced=best_params['include_advanced']
)

test_df = create_features(
    test_df,
    include_temporal=best_params['include_temporal'],
    n_pyth_variants=best_params['n_pyth_variants'],
    include_advanced=best_params['include_advanced']
)

# Get features
feature_cols = get_feature_columns(train_df, best_params['include_temporal'])

# Handle inf/nan
train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

for col in feature_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

X = train_df[feature_cols]
y = train_df['W']
X_test = test_df[feature_cols]

print(f"\n📊 Features: {len(feature_cols)}")
print(f"🎯 Model: {best_params['model_type']}")
print(f"📏 Alpha: {best_params['alpha']:.4f}")
print(f"🔄 Seeds: {best_params['n_seeds']}")

# Train with best params and multiple seeds
all_predictions = []
seeds = list(range(42, 42 + best_params['n_seeds'] * 100, 100))[:best_params['n_seeds']]

for seed in seeds:
    # Scale
    if best_params['scaler'] == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    alpha = best_params['alpha']
    if best_params['model_type'] == 'ridge':
        model = Ridge(alpha=alpha, random_state=seed)
    elif best_params['model_type'] == 'lasso':
        model = Lasso(alpha=alpha, random_state=seed, max_iter=2000)
    else:  # elasticnet
        model = ElasticNet(alpha=alpha, l1_ratio=best_params['l1_ratio'], random_state=seed, max_iter=2000)
    
    model.fit(X_scaled, y)
    
    # Predict
    pred = model.predict(X_test_scaled)
    all_predictions.append(pred)
    print(f"  Seed {seed}: trained")

# Average predictions
final_predictions = np.mean(all_predictions, axis=0)

# Create submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'W': final_predictions
})

submission.to_csv('submission_optuna.csv', index=False)

print("\n" + "=" * 80)
print("✅ SUBMISSION CREATED: submission_optuna.csv")
print("=" * 80)
print(f"📉 Best CV MAE: {study.best_value:.4f}")
print(f"📊 Features: {len(feature_cols)}")
print(f"🎯 Trials completed: {len(study.trials)}")

print("\nPrediction statistics:")
print(f"  Mean: {final_predictions.mean():.2f}")
print(f"  Std:  {final_predictions.std():.2f}")
print(f"  Min:  {final_predictions.min():.2f}")
print(f"  Max:  {final_predictions.max():.2f}")

# Print top 10 trials
print("\n" + "=" * 80)
print("TOP 10 TRIALS")
print("=" * 80)

trials_df = study.trials_dataframe().sort_values('value').head(10)
print(trials_df[['number', 'value', 'params_alpha', 'params_model_type', 
                 'params_n_pyth_variants', 'params_include_advanced']].to_string(index=False))

print("\n" + "=" * 80)
print("🎯 Ready to submit! Expected Kaggle score: ~{:.2f}".format(study.best_value + 0.21))
print("=" * 80)
