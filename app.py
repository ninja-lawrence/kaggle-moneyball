import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost imported successfully")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM imported successfully")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✓ Optuna imported successfully")
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce output noise
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna not available. Install with: pip install optuna")

# Load datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"\nColumn names in train: {train_df.columns.tolist()}")
print(f"\nData types:\n{train_df.dtypes}")
print(f"\nFirst few rows:\n{train_df.head(2)}")

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def create_advanced_features(df):
    """Create advanced baseball analytics features"""
    df = df.copy()
    
    # Helper function to safely get column or return 0
    def safe_get(df, col):
        return df[col] if col in df.columns else 0
    
    # Check required columns
    required_cols = ['G', 'R', 'RA', 'AB', 'H', 'BB', 'SO', 'HR']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return df
    
    # Create derived features that might be missing
    if 'R_per_game' not in df.columns:
        df['R_per_game'] = df['R'] / df['G']
    if 'RA_per_game' not in df.columns:
        df['RA_per_game'] = df['RA'] / df['G']
    
    # 1. PYTHAGOREAN EXPECTATION (Key baseball metric)
    df['pyth_wins'] = (df['R']**2 / (df['R']**2 + df['RA']**2)) * df['G']
    if 'W' in df.columns:
        df['pyth_diff'] = df['W'] - df['pyth_wins']
    
    # Pythagorean with exponent optimization (often 1.83 is better than 2)
    df['pyth_wins_183'] = (df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83)) * df['G']
    
    # 2. RUN DIFFERENTIAL (Most important predictor)
    df['run_diff'] = df['R'] - df['RA']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['run_diff_squared'] = df['run_diff']**2
    
    # 3. OFFENSIVE METRICS
    df['batting_avg'] = df['H'] / df['AB']
    
    # On-base percentage (simplified without HBP and SF)
    df['on_base_pct'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    
    # Slugging (check if we have 2B, 3B, HR)
    if '2B' in df.columns and '3B' in df.columns:
        df['slugging_pct'] = (df['H'] + df['2B'] + 2*df['3B'] + 3*df['HR']) / df['AB']
    else:
        # Simplified slugging without extra base hits detail
        df['slugging_pct'] = (df['H'] + 2*df['HR']) / df['AB']
    
    df['ops'] = df['on_base_pct'] + df['slugging_pct']
    df['iso_power'] = df['slugging_pct'] - df['batting_avg']
    df['bb_so_ratio'] = df['BB'] / (df['SO'] + 1)
    
    if '2B' in df.columns and '3B' in df.columns:
        df['xbh_rate'] = (df['2B'] + df['3B'] + df['HR']) / df['AB']
    else:
        df['xbh_rate'] = df['HR'] / df['AB']
    
    df['hr_rate'] = df['HR'] / df['AB']
    
    # Secondary contact metrics
    df['contact_rate'] = (df['AB'] - df['SO']) / df['AB']
    df['walk_rate'] = df['BB'] / df['AB']
    
    # 4. DEFENSIVE/PITCHING METRICS
    if 'IPouts' in df.columns and 'HA' in df.columns and 'BBA' in df.columns:
        df['whip'] = (df['HA'] + df['BBA']) / (df['IPouts'] / 3 + 0.001)
        df['k_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 0.001)
        df['bb_per_9'] = (df['BBA'] * 27) / (df['IPouts'] + 0.001)
        
        if 'HRA' in df.columns:
            df['hr_per_9'] = (df['HRA'] * 27) / (df['IPouts'] + 0.001)
            df['fip_like'] = ((13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / (df['IPouts']/3 + 0.001))
        
        if 'SOA' in df.columns and 'BBA' in df.columns:
            df['k_bb_ratio'] = df['SOA'] / (df['BBA'] + 1)
        
        if 'HA' in df.columns:
            df['opp_batting_avg'] = df['HA'] / (df['IPouts'] * 3 + 0.001)
    
    # 5. BASERUNNING EFFICIENCY
    if 'SB' in df.columns:
        cs = safe_get(df, 'CS')
        df['sb_success_rate'] = df['SB'] / (df['SB'] + cs + 1)
        df['sb_attempts'] = df['SB'] + cs
        df['sb_per_game'] = df['SB'] / df['G']
    
    # 6. DEFENSIVE EFFICIENCY
    if 'E' in df.columns and 'IPouts' in df.columns:
        df['def_efficiency'] = 1 - (df['E'] / (df['IPouts'] + df['E'] + 0.001))
        df['error_rate'] = df['E'] / df['G']
    
    if 'DP' in df.columns:
        df['double_play_rate'] = df['DP'] / df['G']
    
    # 7. PARK FACTORS & CONTEXT (check if BPF and PPF exist)
    if 'BPF' in df.columns and 'PPF' in df.columns:
        df['park_factor_avg'] = (df['BPF'] + df['PPF']) / 2
        df['park_advantage'] = df['BPF'] - 100
    
    if 'attendance' in df.columns:
        df['attendance_per_game'] = df['attendance'] / df['G']
    
    # 8. GAME SITUATION METRICS
    if 'CG' in df.columns:
        df['complete_game_rate'] = df['CG'] / df['G']
    if 'SHO' in df.columns:
        df['shutout_rate'] = df['SHO'] / df['G']
    if 'SV' in df.columns:
        df['save_rate'] = df['SV'] / df['G']
        if 'W' in df.columns:
            df['save_per_win'] = df['SV'] / (df['W'] + 1)
        else:
            df['save_per_win'] = df['SV'] / df['G']
    
    # 9. RUN PRODUCTION EFFICIENCY
    df['runs_per_hit'] = df['R'] / (df['H'] + 1)
    df['runs_per_homer'] = df['R'] / (df['HR'] + 1)
    
    if '2B' in df.columns and '3B' in df.columns:
        df['runs_per_xbh'] = df['R'] / (df['2B'] + df['3B'] + df['HR'] + 1)
    
    # 10. ERA-ADJUSTED METRICS
    if 'mlb_rpg' in df.columns:
        df['runs_vs_league'] = df['R_per_game'] - df['mlb_rpg']
        df['ra_vs_league'] = df['RA_per_game'] - df['mlb_rpg']
        df['era_adjusted_diff'] = df['runs_vs_league'] - df['ra_vs_league']
        df['r_index'] = (df['R_per_game'] / (df['mlb_rpg'] + 0.001)) * 100
        df['ra_index'] = (df['RA_per_game'] / (df['mlb_rpg'] + 0.001)) * 100
    
    # 11. INTERACTION FEATURES
    df['offense_defense_product'] = df['R'] * (1 / (df['RA'] + 1))
    
    if 'SB' in df.columns and 'sb_success_rate' in df.columns:
        df['ops_x_sb'] = df['ops'] * df['sb_success_rate']
        df['power_speed'] = np.sqrt(df['HR'] * df['SB'])
    
    if 'def_efficiency' in df.columns:
        df['ops_x_def'] = df['ops'] * df['def_efficiency']
    
    if 'whip' in df.columns and 'ERA' in df.columns:
        df['whip_x_era'] = df['whip'] * df['ERA']
    
    if 'ERA' in df.columns:
        df['run_support'] = df['R_per_game'] * df['ERA']
    
    df['quality_contact'] = df['xbh_rate'] * df['ops']
    
    # 12. RATIO FEATURES
    df['r_ra_ratio'] = df['R'] / (df['RA'] + 1)
    df['h_ha_ratio'] = df['H'] / (safe_get(df, 'HA') + 1)
    df['so_soa_ratio'] = df['SO'] / (safe_get(df, 'SOA') + 1)
    df['bb_bba_ratio'] = df['BB'] / (safe_get(df, 'BBA') + 1)
    df['hr_hra_ratio'] = df['HR'] / (safe_get(df, 'HRA') + 1)
    
    # 13. POLYNOMIAL FEATURES (for key metrics)
    df['ops_squared'] = df['ops']**2
    if 'ERA' in df.columns:
        df['era_squared'] = df['ERA']**2
    if 'whip' in df.columns:
        df['whip_squared'] = df['whip']**2
    
    # Replace inf and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

# Apply feature engineering
print("\n" + "="*80)
print("APPLYING ADVANCED FEATURE ENGINEERING")
print("="*80)

# First, check what columns we actually have
print(f"\nColumns available: {len(train_df.columns)}")
print(f"Numeric columns: {len(train_df.select_dtypes(include=[np.number]).columns)}")

train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

# Remove duplicate columns if any
train_df = train_df.loc[:, ~train_df.columns.duplicated()]
test_df = test_df.loc[:, ~test_df.columns.duplicated()]

print(f"\nAfter feature engineering:")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Select features
default_features = [
    'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',
    'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
    'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF',
    'R_per_game', 'RA_per_game', 'mlb_rpg'
]

engineered_features = [
    'pyth_wins', 'pyth_wins_183', 'run_diff', 'run_diff_per_game', 'run_diff_squared',
    'batting_avg', 'on_base_pct', 'slugging_pct', 'ops', 'iso_power',
    'bb_so_ratio', 'xbh_rate', 'hr_rate', 'contact_rate', 'walk_rate',
    'whip', 'k_per_9', 'bb_per_9', 'hr_per_9', 'k_bb_ratio', 'opp_batting_avg', 'fip_like',
    'sb_success_rate', 'sb_attempts', 'sb_per_game',
    'def_efficiency', 'double_play_rate', 'error_rate',
    'park_factor_avg', 'park_advantage', 'attendance_per_game',
    'complete_game_rate', 'shutout_rate', 'save_rate', 'save_per_win',
    'runs_per_hit', 'runs_per_homer', 'runs_per_xbh',
    'runs_vs_league', 'ra_vs_league', 'era_adjusted_diff', 'r_index', 'ra_index',
    'offense_defense_product', 'ops_x_sb', 'power_speed', 'ops_x_def', 'whip_x_era',
    'run_support', 'quality_contact',
    'r_ra_ratio', 'h_ha_ratio', 'so_soa_ratio', 'bb_bba_ratio', 'hr_hra_ratio',
    'ops_squared', 'era_squared', 'whip_squared'
]

era_features = [col for col in train_df.columns if col.startswith('era_')]
decade_features = [col for col in train_df.columns if col.startswith('decade_')]

all_features = default_features + engineered_features + era_features + decade_features
# Remove duplicates from feature list (keep first occurrence)
seen = set()
available_features = []
for col in all_features:
    if col in train_df.columns and col in test_df.columns and col not in seen:
        available_features.append(col)
        seen.add(col)

print(f"Total features: {len(available_features)}")

# Prepare data
X_train = train_df[available_features]
y_train = train_df['W']
X_test = test_df[available_features]
# Note: test set doesn't have 'W' column - that's what we're predicting

# Save the ID column from test set AFTER all processing for submission
# This ensures the IDs match the exact rows we're making predictions for
if 'ID' in test_df.columns:
    test_ids = test_df['ID'].copy()
    print(f"\n✓ Found ID column in test set")
    print(f"  First few IDs: {test_ids.head().tolist()}")
    print(f"  Total test IDs: {len(test_ids)}")
else:
    # Fallback: use pandas index if ID column doesn't exist
    test_ids = test_df.index.copy()
    print(f"\n⚠ No ID column found, using pandas index")

print(f"\nFeature matrix shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"\nChecking for any remaining NaN values:")
print(f"X_train NaN count: {X_train.isna().sum().sum()}")
print(f"X_test NaN count: {X_test.isna().sum().sum()}")

# If there are any NaN values, fill them
if X_train.isna().sum().sum() > 0:
    print("Filling remaining NaN values with median...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test

# ============================================================================
# FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("SCALING FEATURES")
print("="*80)

one_hot_cols = [col for col in X_train.columns if col.startswith(('era_', 'decade_'))]
scale_cols = [col for col in X_train.columns if col not in one_hot_cols]

scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test=None):
    """Evaluate model performance using cross-validation on training data"""
    # Convert to numpy arrays if DataFrame (XGBoost has issues with some DataFrame formats)
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Use 5-fold CV for robust evaluation
    cv_scores = cross_val_score(model, X_train_arr, y_train_arr, cv=5, 
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Fit on full training set and get predictions
    model.fit(X_train_arr, y_train_arr)
    train_pred = model.predict(X_train_arr)
    test_pred = model.predict(X_test_arr)
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    return {
        'train_mae': train_mae,
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'train_r2': train_r2,
        'train_pred': train_pred,
        'test_pred': test_pred
    }

# ============================================================================
# AUTOMATED HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================

# Initialize results and models dictionaries
models = {}
results = {}

if OPTUNA_AVAILABLE:
    print("\n" + "="*80)
    print("AUTOMATED HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80)
    
    # Split data for validation
    from sklearn.model_selection import train_test_split
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Convert to numpy arrays to avoid XGBoost DataFrame issues
    X_train_opt = X_train_opt.values if hasattr(X_train_opt, 'values') else X_train_opt
    X_val_opt = X_val_opt.values if hasattr(X_val_opt, 'values') else X_val_opt
    y_train_opt = y_train_opt.values if hasattr(y_train_opt, 'values') else y_train_opt
    y_val_opt = y_val_opt.values if hasattr(y_val_opt, 'values') else y_val_opt
    
    # 1. XGBoost Optimization
    if XGBOOST_AVAILABLE:
        print("\n1. Optimizing XGBoost hyperparameters...")
        
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_opt, y_train_opt)
            preds = model.predict(X_val_opt)
            mae = mean_absolute_error(y_val_opt, preds)
            return mae
        
        study_xgb = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
        study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)
        
        print(f"   Best MAE: {study_xgb.best_value:.4f}")
        print(f"   Best params: {study_xgb.best_params}")
        
        # Train final model with best params
        xgb_optimized = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1, verbosity=0)
        models['XGBoost (Optimized)'] = xgb_optimized
        results['XGBoost (Optimized)'] = evaluate_model(xgb_optimized, X_train_scaled, y_train, X_test_scaled)
        print(f"   CV MAE: {results['XGBoost (Optimized)']['cv_mae']:.4f}")
    
    # 2. LightGBM Optimization
    if LIGHTGBM_AVAILABLE:
        print("\n2. Optimizing LightGBM hyperparameters...")
        
        def objective_lgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_opt, y_train_opt)
            preds = model.predict(X_val_opt)
            mae = mean_absolute_error(y_val_opt, preds)
            return mae
        
        study_lgb = optuna.create_study(direction='minimize', study_name='lightgbm_optimization')
        study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=False)
        
        print(f"   Best MAE: {study_lgb.best_value:.4f}")
        print(f"   Best params: {study_lgb.best_params}")
        
        # Train final model with best params
        lgb_optimized = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, n_jobs=-1, verbose=-1)
        lgb_optimized.fit(X_train_scaled, y_train)
        models['LightGBM (Optimized)'] = lgb_optimized
        results['LightGBM (Optimized)'] = evaluate_model(lgb_optimized, X_train_scaled, y_train,
                                                          X_test_scaled)
        print(f"   CV MAE: {results['LightGBM (Optimized)']['cv_mae']:.4f}")
    
    # 3. Random Forest Optimization
    print("\n3. Optimizing Random Forest hyperparameters...")
    
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train_opt, y_train_opt)
        preds = model.predict(X_val_opt)
        mae = mean_absolute_error(y_val_opt, preds)
        return mae
    
    study_rf = optuna.create_study(direction='minimize', study_name='rf_optimization')
    study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=False)
    
    print(f"   Best MAE: {study_rf.best_value:.4f}")
    print(f"   Best params: {study_rf.best_params}")
    
    # Train final model with best params
    rf_optimized = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
    rf_optimized.fit(X_train_scaled, y_train)
    models['Random Forest (Optimized)'] = rf_optimized
    results['Random Forest (Optimized)'] = evaluate_model(rf_optimized, X_train_scaled, y_train,
                                                           X_test_scaled)
    print(f"   CV MAE: {results['Random Forest (Optimized)']['cv_mae']:.4f}")
    
    # 4. Gradient Boosting Optimization
    print("\n4. Optimizing Gradient Boosting hyperparameters...")
    
    def objective_gb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train_opt, y_train_opt)
        preds = model.predict(X_val_opt)
        mae = mean_absolute_error(y_val_opt, preds)
        return mae
    
    study_gb = optuna.create_study(direction='minimize', study_name='gb_optimization')
    study_gb.optimize(objective_gb, n_trials=30, show_progress_bar=False)
    
    print(f"   Best MAE: {study_gb.best_value:.4f}")
    print(f"   Best params: {study_gb.best_params}")
    
    # Train final model with best params
    gb_optimized = GradientBoostingRegressor(**study_gb.best_params, random_state=42)
    gb_optimized.fit(X_train_scaled, y_train)
    models['Gradient Boosting (Optimized)'] = gb_optimized
    results['Gradient Boosting (Optimized)'] = evaluate_model(gb_optimized, X_train_scaled, y_train,
                                                               X_test_scaled)
    print(f"   CV MAE: {results['Gradient Boosting (Optimized)']['cv_mae']:.4f}")
    
    # 5. Create Super Ensemble with Optimized Models
    print("\n5. Creating Super Ensemble with Optimized Models...")
    
    super_estimators = [
        ('ridge', Ridge(alpha=10.0)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
    ]
    
    if XGBOOST_AVAILABLE:
        super_estimators.append(('xgb_opt', xgb_optimized))
    if LIGHTGBM_AVAILABLE:
        super_estimators.append(('lgb_opt', lgb_optimized))
    
    super_estimators.extend([
        ('rf_opt', rf_optimized),
        ('gb_opt', gb_optimized)
    ])
    
    # Determine best meta-learner
    if XGBOOST_AVAILABLE:
        meta_learner = xgb.XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3,
                                        random_state=42, verbosity=0)
    elif LIGHTGBM_AVAILABLE:
        meta_learner = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3,
                                         random_state=42, verbose=-1)
    else:
        meta_learner = Ridge(alpha=1.0)
    
    super_stack = StackingRegressor(
        estimators=super_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    super_stack.fit(X_train_scaled, y_train)
    models['Super Ensemble (Optimized)'] = super_stack
    results['Super Ensemble (Optimized)'] = evaluate_model(super_stack, X_train_scaled, y_train,
                                                            X_test_scaled)
    print(f"   CV MAE: {results['Super Ensemble (Optimized)']['cv_mae']:.4f}")
    
    print("\n✨ Optimization Complete!")

else:
    print("\n" + "="*80)
    print("OPTUNA NOT AVAILABLE - Skipping Automated Hyperparameter Tuning")
    print("="*80)
    print("Install Optuna for automated optimization: pip install optuna")

# ============================================================================
# BASE MODELS WITH OPTIMIZED HYPERPARAMETERS (if Optuna not available)
# ============================================================================
print("\n" + "="*80)
print("TRAINING BASE MODELS")
print("="*80)

models = {}
results = {}

# 1. Ridge Regression
print("\n1. Ridge Regression")
ridge = Ridge(alpha=10.0)
models['Ridge'] = ridge
results['Ridge'] = evaluate_model(ridge, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['Ridge']['cv_mae']:.4f} ± {results['Ridge']['cv_std']:.4f}")

# 2. Lasso Regression
print("\n2. Lasso Regression")
lasso = Lasso(alpha=0.1, max_iter=10000)
models['Lasso'] = lasso
results['Lasso'] = evaluate_model(lasso, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['Lasso']['cv_mae']:.4f} ± {results['Lasso']['cv_std']:.4f}")

# 3. ElasticNet
print("\n3. ElasticNet")
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
models['ElasticNet'] = elastic
results['ElasticNet'] = evaluate_model(elastic, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['ElasticNet']['cv_mae']:.4f} ± {results['ElasticNet']['cv_std']:.4f}")

# 4. Random Forest
print("\n4. Random Forest")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, 
                           min_samples_leaf=1, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf
results['Random Forest'] = evaluate_model(rf, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['Random Forest']['cv_mae']:.4f}")

# 5. Gradient Boosting
print("\n5. Gradient Boosting")
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                                min_samples_split=2, random_state=42)
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb
results['Gradient Boosting'] = evaluate_model(gb, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['Gradient Boosting']['cv_mae']:.4f}")

# 6. XGBoost (if available)
if XGBOOST_AVAILABLE:
    print("\n6. XGBoost")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = evaluate_model(xgb_model, X_train_scaled, y_train, X_test_scaled)
    print(f"   CV MAE: {results['XGBoost']['cv_mae']:.4f}")
else:
    print("\n6. XGBoost - Skipped (not installed)")

# 7. LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    print("\n7. LightGBM")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    models['LightGBM'] = lgb_model
    results['LightGBM'] = evaluate_model(lgb_model, X_train_scaled, y_train, X_test_scaled)
    print(f"   CV MAE: {results['LightGBM']['cv_mae']:.4f}")
else:
    print("\n7. LightGBM - Skipped (not installed)")

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================
print("\n" + "="*80)
print("CREATING ENSEMBLE MODELS")
print("="*80)

# 1. VOTING REGRESSOR (Simple Average)
print("\n1. Voting Regressor (Weighted Average)")

# Build estimators list dynamically based on availability
voting_estimators = [
    ('ridge', Ridge(alpha=10.0)),
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
]
voting_weights = [1, 1, 2, 2]

if XGBOOST_AVAILABLE:
    voting_estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                                      random_state=42, n_jobs=-1, verbosity=0)))
    voting_weights.append(2)

if LIGHTGBM_AVAILABLE:
    voting_estimators.append(('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                                       random_state=42, n_jobs=-1, verbose=-1)))
    voting_weights.append(2)

voting = VotingRegressor(estimators=voting_estimators, weights=voting_weights)
voting.fit(X_train_scaled, y_train)
models['Voting Ensemble'] = voting
results['Voting Ensemble'] = evaluate_model(voting, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['Voting Ensemble']['cv_mae']:.4f}")

# 2. STACKING REGRESSOR (Meta-Model)
print("\n2. Stacking Regressor (Multi-Layer Stack)")

# Build base models dynamically
base_models = [
    ('ridge', Ridge(alpha=10.0)),
    ('lasso', Lasso(alpha=0.1, max_iter=10000)),
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=2, 
                                 random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                     min_samples_split=2, random_state=42))
]

if XGBOOST_AVAILABLE:
    base_models.append(('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                                 random_state=42, n_jobs=-1, verbosity=0)))

if LIGHTGBM_AVAILABLE:
    base_models.append(('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                                  random_state=42, n_jobs=-1, verbose=-1)))

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

print("   Training stacking model (this may take a minute)...")
stacking.fit(X_train_scaled, y_train)
models['Stacking Ensemble'] = stacking
results['Stacking Ensemble'] = evaluate_model(stacking, X_train_scaled, y_train, X_test_scaled)
print(f"   CV MAE: {results['Stacking Ensemble']['cv_mae']:.4f}")

# 3. ADVANCED STACKING with XGBoost/LightGBM Meta-Learner
if XGBOOST_AVAILABLE:
    print("\n3. Advanced Stacking (XGBoost Meta-Model)")
    stacking_xgb = StackingRegressor(
        estimators=base_models,
        final_estimator=xgb.XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3,
                                         random_state=42, verbosity=0),
        cv=5,
        n_jobs=-1
    )
    print("   Training XGBoost stacking model...")
    stacking_xgb.fit(X_train_scaled, y_train)
    models['Stacking (XGB Meta)'] = stacking_xgb
    results['Stacking (XGB Meta)'] = evaluate_model(stacking_xgb, X_train_scaled, y_train, X_test_scaled)
    print(f"   CV MAE: {results['Stacking (XGB Meta)']['cv_mae']:.4f}")

if LIGHTGBM_AVAILABLE:
    print("\n4. Advanced Stacking (LightGBM Meta-Model)")
    stacking_lgb = StackingRegressor(
        estimators=base_models,
        final_estimator=lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3,
                                          random_state=42, verbose=-1),
        cv=5,
        n_jobs=-1
    )
    print("   Training LightGBM stacking model...")
    stacking_lgb.fit(X_train_scaled, y_train)
    models['Stacking (LGB Meta)'] = stacking_lgb
    results['Stacking (LGB Meta)'] = evaluate_model(stacking_lgb, X_train_scaled, y_train, X_test_scaled)
    print(f"   CV MAE: {results['Stacking (LGB Meta)']['cv_mae']:.4f}")

# 5. BLENDING (Holdout-based ensemble)
print(f"\n5. Blending (Holdout-based Ensemble)")

from sklearn.model_selection import train_test_split
X_blend_train, X_blend_holdout, y_blend_train, y_blend_holdout = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Build blend models dynamically
blend_models = {
    'ridge': Ridge(alpha=10.0),
    'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    'rf': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    'gb': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

if XGBOOST_AVAILABLE:
    blend_models['xgb'] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                           random_state=42, n_jobs=-1, verbosity=0)

if LIGHTGBM_AVAILABLE:
    blend_models['lgb'] = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                            random_state=42, n_jobs=-1, verbose=-1)

# Get predictions on holdout set
blend_holdout_preds = np.zeros((len(X_blend_holdout), len(blend_models)))
blend_test_preds = np.zeros((len(X_test_scaled), len(blend_models)))

for i, (name, model) in enumerate(blend_models.items()):
    model.fit(X_blend_train, y_blend_train)
    blend_holdout_preds[:, i] = model.predict(X_blend_holdout)
    blend_test_preds[:, i] = model.predict(X_test_scaled)

# Train meta-model on holdout predictions
meta_model = Ridge(alpha=1.0)
meta_model.fit(blend_holdout_preds, y_blend_holdout)

# Final predictions
blend_final_test = meta_model.predict(blend_test_preds)

# For blending, we'll use the holdout CV MAE as our estimate
blend_cv_mae = mean_absolute_error(y_blend_holdout, meta_model.predict(blend_holdout_preds))

results['Blending'] = {
    'train_mae': 0,
    'cv_mae': blend_cv_mae,
    'cv_std': 0,  # Not computed for blending
    'train_r2': 0,
    'test_pred': blend_final_test
}

print(f"   CV MAE: {blend_cv_mae:.4f}")

# ============================================================================
# RESULTS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
print("="*80)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train MAE': [results[m]['train_mae'] for m in results.keys()],
    'CV MAE': [results[m]['cv_mae'] for m in results.keys()],
    'CV Std': [results[m]['cv_std'] for m in results.keys()],
    'Train R²': [results[m]['train_r2'] for m in results.keys()]
})

results_df = results_df.sort_values('CV MAE')
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = models.get(best_model_name, None)

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   CV MAE: {results_df.iloc[0]['CV MAE']:.4f} ± {results_df.iloc[0]['CV Std']:.4f}")
print(f"   Train MAE: {results_df.iloc[0]['Train MAE']:.4f}")
print(f"   Train R²: {results_df.iloc[0]['Train R²']:.4f}")

# Show improvement over baseline
baseline_mae = results['Ridge']['cv_mae']
best_mae = results_df.iloc[0]['CV MAE']
improvement = ((baseline_mae - best_mae) / baseline_mae) * 100
print(f"\n   Improvement over Ridge baseline: {improvement:.2f}%")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Try to get feature importance from best ensemble model
if best_model_name == 'Stacking Ensemble' or best_model_name == 'Stacking (GB Meta)':
    # Average importance from base estimators
    importances_list = []
    for name, estimator in best_model.named_estimators_.items():
        if hasattr(estimator, 'feature_importances_'):
            importances_list.append(estimator.feature_importances_)
        elif hasattr(estimator, 'coef_'):
            importances_list.append(np.abs(estimator.coef_))
    
    if importances_list:
        importances = np.mean(importances_list, axis=0)
    else:
        importances = np.abs(ridge.coef_)  # Fallback
        
elif best_model_name == 'Voting Ensemble':
    # Use Random Forest importance as proxy
    importances = rf.feature_importances_
else:
    # Use the model's own importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_) if hasattr(best_model, 'coef_') else np.abs(ridge.coef_)

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Model Comparison - MAE
ax1 = fig.add_subplot(gs[0, 0])
results_sorted = results_df.sort_values('CV MAE', ascending=True)
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_sorted))]
ax1.barh(range(len(results_sorted)), results_sorted['CV MAE'], color=colors)
ax1.set_yticks(range(len(results_sorted)))
ax1.set_yticklabels(results_sorted['Model'])
ax1.set_xlabel('CV MAE (Lower is Better)')
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.axvline(x=baseline_mae, color='r', linestyle='--', alpha=0.5, label='Baseline (Ridge)')
ax1.legend()

# 2. Actual vs Predicted (Best Model - on training data)
ax2 = fig.add_subplot(gs[0, 1])
best_train_preds = results[best_model_name]['train_pred']
ax2.scatter(y_train, best_train_preds, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Wins')
ax2.set_ylabel('Predicted Wins')
ax2.set_title(f'{best_model_name}: Actual vs Predicted (Train)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
r2 = results[best_model_name]['train_r2']
ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Residual Plot
ax3 = fig.add_subplot(gs[0, 2])
residuals = y_train - best_train_preds
ax3.scatter(best_train_preds, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Wins')
ax3.set_ylabel('Residuals')
ax3.set_title(f'{best_model_name}: Residual Plot', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Feature Importance (Top 15)
ax4 = fig.add_subplot(gs[1, 0])
top_features = feature_importance.head(15)
colors_feat = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax4.barh(range(len(top_features)), top_features['Importance'], color=colors_feat)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'])
ax4.set_xlabel('Importance Score')
ax4.set_title('Top 15 Feature Importances', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Error Distribution
ax5 = fig.add_subplot(gs[1, 1])
errors = np.abs(residuals)
ax5.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax5.set_xlabel('Absolute Error (Wins)')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Error Distribution', fontweight='bold')
ax5.axvline(x=errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
ax5.axvline(x=errors.median(), color='g', linestyle='--', linewidth=2, label=f'Median: {errors.median():.2f}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. R² Comparison
ax6 = fig.add_subplot(gs[1, 2])
results_r2_sorted = results_df.sort_values('Train R²', ascending=True)
colors_r2 = ['#2ecc71' if i == len(results_r2_sorted)-1 else '#3498db' for i in range(len(results_r2_sorted))]
ax6.barh(range(len(results_r2_sorted)), results_r2_sorted['Train R²'], color=colors_r2)
ax6.set_yticks(range(len(results_r2_sorted)))
ax6.set_yticklabels(results_r2_sorted['Model'])
ax6.set_xlabel('Train R² (Higher is Better)')
ax6.set_title('Model R² Comparison', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# 7. Ensemble Comparison
ax7 = fig.add_subplot(gs[2, 0])
ensemble_model_names = ['Ridge', 'Gradient Boosting']
if XGBOOST_AVAILABLE:
    ensemble_model_names.append('XGBoost')
if LIGHTGBM_AVAILABLE:
    ensemble_model_names.append('LightGBM')
ensemble_model_names.extend(['Voting Ensemble', 'Stacking Ensemble', 'Blending'])

ensemble_maes = [results[m]['cv_mae'] for m in ensemble_model_names if m in results]
ensemble_names = [m for m in ensemble_model_names if m in results]
x_pos = np.arange(len(ensemble_names))

# Color code: base models in blue, ensembles in green
colors_ens = []
for name in ensemble_names:
    if 'Ensemble' in name or 'Blending' in name:
        colors_ens.append('#27ae60')
    elif 'XGBoost' in name or 'LightGBM' in name:
        colors_ens.append('#f39c12')
    else:
        colors_ens.append('#3498db')

bars = ax7.bar(x_pos, ensemble_maes, color=colors_ens, alpha=0.7, edgecolor='black')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(ensemble_names, rotation=45, ha='right', fontsize=9)
ax7.set_ylabel('CV MAE')
ax7.set_title('Base Models vs Ensemble Methods', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 8. Cross-validation scores for top 3 models
ax8 = fig.add_subplot(gs[2, 1])
top_3_models = results_df.head(3)['Model'].tolist()
cv_scores = []
cv_labels = []

for model_name in top_3_models:
    if model_name in models and models[model_name] is not None:
        model = models[model_name]
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_scores.append(-scores)
        cv_labels.append(model_name)

bp = ax8.boxplot(cv_scores, labels=cv_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax8.set_ylabel('Cross-Validation MAE')
ax8.set_title('5-Fold CV Performance (Top 3 Models)', fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 9. Learning Curve (Best Model)
ax9 = fig.add_subplot(gs[2, 2])
train_sizes = np.linspace(0.1, 1.0, 10)
train_errors = []
test_errors = []

for size in train_sizes:
    n_samples = int(len(X_train_scaled) * size)
    X_subset = X_train_scaled[:n_samples]
    y_subset = y_train[:n_samples]
    
    # Use a simple model with CV for validation error
    temp_model = Ridge(alpha=10.0)
    temp_model.fit(X_subset, y_subset)
    
    train_pred = temp_model.predict(X_subset)
    train_errors.append(mean_absolute_error(y_subset, train_pred))
    
    # Use CV for validation error
    cv_scores = cross_val_score(temp_model, X_subset, y_subset, cv=5, 
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
    test_errors.append(-cv_scores.mean())

ax9.plot(train_sizes * 100, train_errors, 'o-', label='Training Error', linewidth=2, markersize=6)
ax9.plot(train_sizes * 100, test_errors, 's-', label='CV Error', linewidth=2, markersize=6)
ax9.set_xlabel('Training Set Size (%)')
ax9.set_ylabel('Mean Absolute Error')
ax9.set_title('Learning Curve', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("FINAL ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 BEST PERFORMING MODEL: {best_model_name}")
print(f"   • CV MAE: {results[best_model_name]['cv_mae']:.4f} ± {results[best_model_name]['cv_std']:.4f} wins")
print(f"   • Train MAE: {results[best_model_name]['train_mae']:.4f} wins")
print(f"   • Train R²: {results[best_model_name]['train_r2']:.4f}")
print(f"   • Improvement over baseline: {improvement:.2f}%")

print("\n🎯 KEY INSIGHTS:")
print(f"   1. Top 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"      • {row['Feature']}: {row['Importance']:.4f}")

print(f"\n   2. Average Prediction Error: {results[best_model_name]['cv_mae']:.2f} wins")
print(f"   3. 95% of predictions within: ±{np.percentile(np.abs(residuals), 95):.2f} wins")

print("\n🔍 ENSEMBLE PERFORMANCE:")
ensemble_results = results_df[results_df['Model'].str.contains('Ensemble|Blending')]
if len(ensemble_results) > 0:
    for _, row in ensemble_results.iterrows():
        print(f"   • {row['Model']}: MAE = {row['CV MAE']:.4f}")

print("\n💡 RECOMMENDATIONS:")
print("   1. Pythagorean wins and run differential are likely the strongest predictors")
print("   2. Ensemble methods (especially stacking) provide robust predictions")
if XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE:
    print("   3. XGBoost/LightGBM often outperform traditional methods on tabular data")
    print("   4. Consider using the best ensemble model for final predictions")
else:
    print("   3. Consider installing XGBoost/LightGBM for potentially better performance:")
    print("      pip install xgboost lightgbm")
    print("   4. Use the best ensemble model for final predictions")
print("   5. Monitor residual patterns for systematic biases")

# Model recommendations based on what's available
print("\n🎯 MODEL RECOMMENDATIONS:")
if OPTUNA_AVAILABLE:
    print("   ✓ Optuna optimization enabled - models are hyperparameter-tuned")
    print("   → Recommended: Super Ensemble (Optimized) for best performance")
    
    # Show optimization improvements
    if 'XGBoost' in results and 'XGBoost (Optimized)' in results:
        improvement_xgb = ((results['XGBoost']['cv_mae'] - results['XGBoost (Optimized)']['cv_mae']) 
                          / results['XGBoost']['cv_mae'] * 100)
        print(f"   → XGBoost improvement: {improvement_xgb:.2f}%")
    
    if 'LightGBM' in results and 'LightGBM (Optimized)' in results:
        improvement_lgb = ((results['LightGBM']['cv_mae'] - results['LightGBM (Optimized)']['cv_mae'])
                          / results['LightGBM']['cv_mae'] * 100)
        print(f"   → LightGBM improvement: {improvement_lgb:.2f}%")

elif XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE:
    print("   ✓ Full suite available: XGBoost, LightGBM, and all ensemble methods")
    print("   → Recommended: Stacking with XGBoost/LightGBM meta-learner")
    print("   💡 Install Optuna for auto-tuning: pip install optuna")
elif XGBOOST_AVAILABLE:
    print("   ✓ XGBoost available for enhanced performance")
    print("   → Recommended: Stacking with XGBoost meta-learner")
    print("   💡 Install Optuna for auto-tuning: pip install optuna")
elif LIGHTGBM_AVAILABLE:
    print("   ✓ LightGBM available for enhanced performance")
    print("   → Recommended: Stacking with LightGBM meta-learner")
    print("   💡 Install Optuna for auto-tuning: pip install optuna")
else:
    print("   → Using sklearn models only (solid baseline)")
    print("   → For best results, install: pip install xgboost lightgbm optuna")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Model ready for Kaggle submission!")
print("="*80)

# Show Optuna optimization summary if available
if OPTUNA_AVAILABLE:
    print("\n📊 OPTUNA OPTIMIZATION SUMMARY:")
    if XGBOOST_AVAILABLE:
        print(f"\n   XGBoost - {len(study_xgb.trials)} trials completed")
        print(f"   • Best trial: #{study_xgb.best_trial.number}")
        print(f"   • Best MAE: {study_xgb.best_value:.4f}")
    
    if LIGHTGBM_AVAILABLE:
        print(f"\n   LightGBM - {len(study_lgb.trials)} trials completed")
        print(f"   • Best trial: #{study_lgb.best_trial.number}")
        print(f"   • Best MAE: {study_lgb.best_value:.4f}")
    
    print(f"\n   Random Forest - {len(study_rf.trials)} trials completed")
    print(f"   • Best trial: #{study_rf.best_trial.number}")
    print(f"   • Best MAE: {study_rf.best_value:.4f}")
    
    print(f"\n   Gradient Boosting - {len(study_gb.trials)} trials completed")
    print(f"   • Best trial: #{study_gb.best_trial.number}")
    print(f"   • Best MAE: {study_gb.best_value:.4f}")

# Optional: Create submission file
# ===================== Kaggle Workflow =====================
# 1. Split train.csv for validation ONLY
# 2. Train/tune model using train/validation sets
# 3. Use test.csv ONLY for final prediction and submission
# 4. Submission file must use IDs from test.csv
# ===========================================================

if best_model is not None:
    print("\n📝 Creating predictions for submission...")
    final_predictions = best_model.predict(X_test_scaled)
    final_predictions = np.clip(final_predictions, 0, 162)
    print(f"   • Mean prediction: {final_predictions.mean():.2f} wins")
    print(f"   • Prediction range: {final_predictions.min():.2f} to {final_predictions.max():.2f} wins")
    print("\n   Use these predictions for your Kaggle submission!")
    # Check alignment and length of IDs and predictions
    print(f"   test_ids length: {len(test_ids)}")
    print(f"   final_predictions length: {len(final_predictions)}")
    if len(test_ids) != len(final_predictions):
        print(f"   ERROR: test_ids and predictions length mismatch!")
        print(f"   test_ids sample: {test_ids[:5]}")
        print(f"   predictions sample: {final_predictions[:5]}")
    else:
        print(f"   test_ids and predictions lengths match.")
    # Convert predictions to integers as required by Kaggle
    final_predictions_int = np.round(final_predictions).astype(int)
    # Create submission file
    submission_df = pd.DataFrame({'ID': test_ids, 'W': final_predictions_int})
    submission_df['ID'] = submission_df['ID'].astype(int)
    # Sort by ID for Kaggle
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    submission_df.to_csv('submission.csv', index=False)
    print("   Saved to: submission.csv")
    print(f"   Submission shape: {submission_df.shape}")
    print(f"   First few rows:\n{submission_df.head()}")
    print(f"   Last few rows:\n{submission_df.tail()}")