"""
Neural Network Approach for Kaggle Moneyball
============================================

Key Design Decisions:
1. Simple architecture to avoid CV overfitting (our nemesis!)
2. Strong regularization (dropout, L2, early stopping)
3. No temporal features (proven to hurt)
4. Moderate feature count (47-51 sweet spot)
5. Conservative training to prevent overfitting

Expected Result: ~3.00-3.02 based on pattern
(Neural networks are prone to CV overfitting)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Install tensorflow if needed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
except ImportError:
    print("Installing tensorflow...")
    import subprocess
    subprocess.run(['pip', 'install', 'tensorflow'], check=True)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

print("=" * 80)
print("NEURAL NETWORK APPROACH")
print("=" * 80)
print("\nDesign Philosophy:")
print("- Simple architecture (avoid complexity)")
print("- Heavy regularization (prevent overfitting)")
print("- Early stopping (stop before CV overfitting)")
print("- No temporal features (test has no yearID)")
print("- Conservative approach based on lessons learned")
print()

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Feature engineering (no temporal features!)
def create_features(df):
    df = df.copy()
    
    # Basic stats (use R for runs scored, G for games)
    if 'G' in df.columns:
        df['total_games'] = df['G']
    
    # Pythagorean expectation variations (proven important)
    if 'R' in df.columns and 'RA' in df.columns:
        for exp in [1.83, 1.85, 1.9, 2.0]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            if 'G' in df.columns:
                df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        # Run differential
        df['run_diff'] = df['R'] - df['RA']
        if 'G' in df.columns:
            df['run_diff_per_game'] = df['run_diff'] / df['G']
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    
    # Offensive stats
    if 'H' in df.columns and 'AB' in df.columns:
        df['BA'] = df['H'] / (df['AB'] + 1)
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            df['OPS'] = df['OBP'] + df['SLG']
            df['ISO'] = df['SLG'] - df['BA']
    
    # Pitching stats
    if 'IPouts' in df.columns:
        if 'HA' in df.columns and 'BBA' in df.columns:
            df['WHIP'] = (df['HA'] + df['BBA']) / ((df['IPouts'] / 3) + 1)
        if 'SOA' in df.columns and 'BBA' in df.columns:
            df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
            df['K_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 1)
        if 'HRA' in df.columns and 'BBA' in df.columns and 'SOA' in df.columns:
            df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / (df['IPouts'] + 1)
    
    # Run production rates
    if 'G' in df.columns:
        for col in ['R', 'RA', 'H', 'HR', 'BB', 'SO']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / df['G']
    
    # Key interactions
    if 'OPS' in df.columns and 'WHIP' in df.columns:
        df['OPS_WHIP'] = df['OPS'] * (1 / df['WHIP'].clip(lower=0.1))
    if 'ISO' in df.columns and 'FIP' in df.columns:
        df['ISO_FIP'] = df['ISO'] * (1 / df['FIP'].clip(lower=0.1))
    
    # Replace inf with nan, then fill
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# Create features
train_df = create_features(train_df)
test_df = create_features(test_df)

# Select features (exclude temporal and identifiers)
exclude_cols = ['W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins', 'mlb_rpg']
exclude_cols += [col for col in train_df.columns if 'decade' in col or 'era' in col]

feature_cols = [col for col in train_df.columns if col not in exclude_cols]
feature_cols = [col for col in feature_cols if col in test_df.columns]

print(f"\nFeature count: {len(feature_cols)}")

X_train = train_df[feature_cols].values
y_train = train_df['W'].values
X_test = test_df[feature_cols].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nInput shape: {X_train_scaled.shape}")
print(f"Output range: {y_train.min():.0f} to {y_train.max():.0f}")

# Build neural network with strong regularization
def build_model(input_dim):
    """
    Simple architecture to minimize overfitting:
    - 2 hidden layers (not deep)
    - Moderate width (64, 32 neurons)
    - Heavy dropout (0.3, 0.2)
    - L2 regularization (0.01)
    - LeakyReLU (avoid dead neurons)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Layer 1: 64 neurons with regularization
        layers.Dense(64, 
                    kernel_regularizer=regularizers.l2(0.01),
                    activation='linear'),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Layer 2: 32 neurons with regularization
        layers.Dense(32,
                    kernel_regularizer=regularizers.l2(0.01),
                    activation='linear'),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use MAE loss since that's what we're scored on
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae']
    )
    
    return model

# Cross-validation with early stopping
print("\n" + "="*80)
print("CROSS-VALIDATION")
print("="*80)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_train_scaled))
test_predictions = np.zeros(len(X_test_scaled))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    print(f"\nFold {fold}/{n_splits}")
    print("-" * 40)
    
    X_tr = X_train_scaled[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_scaled[val_idx]
    y_val = y_train[val_idx]
    
    # Build fresh model
    model = build_model(X_train_scaled.shape[1])
    
    # Early stopping to prevent overfitting
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0
    )
    
    # Train with validation monitoring
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Predict on validation
    val_pred = model.predict(X_val, verbose=0).flatten()
    oof_predictions[val_idx] = val_pred
    
    # Predict on test
    test_pred = model.predict(X_test_scaled, verbose=0).flatten()
    test_predictions += test_pred / n_splits
    
    # Calculate fold score
    fold_mae = mean_absolute_error(y_val, val_pred)
    fold_scores.append(fold_mae)
    
    print(f"Validation MAE: {fold_mae:.4f}")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Final train MAE: {history.history['loss'][-1]:.4f}")
    print(f"Final val MAE: {history.history['val_loss'][-1]:.4f}")

# Overall CV score
cv_mae = mean_absolute_error(y_train, oof_predictions)
cv_std = np.std(fold_scores)

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print(f"CV MAE: {cv_mae:.4f} ± {cv_std:.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")

print("\n" + "="*80)
print("EXPECTED KAGGLE SCORE ANALYSIS")
print("="*80)
print("\nBased on the pattern we discovered:")
print(f"CV MAE: {cv_mae:.4f}")
print(f"\nHistorical CV-to-Kaggle gaps:")
print("  Simple blend:  2.77 → 2.98 (gap: 0.21) ✅ BEST")
print("  Stacking:      2.77 → 3.01 (gap: 0.24)")
print("  Advanced feat: 2.76 → 3.02 (gap: 0.26)")
print("  Optuna:        2.76 → 3.02 (gap: 0.26)")
print("  Improved:      2.72 → 3.01 (gap: 0.29)")
print("  Adversarial:   2.71 → 3.05 (gap: 0.34) ❌ WORST")
print(f"\nNeural networks tend to overfit CV even more...")
print(f"Expected Kaggle: ~{cv_mae + 0.25:.2f} to {cv_mae + 0.30:.2f}")
print(f"\nLikely will NOT beat the simple blend's 2.98")

# Create submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'W': np.clip(test_predictions, 0, 162)  # Clip to valid range
})

submission.to_csv('submission_neural_network.csv', index=False)

print("\n" + "="*80)
print("SUBMISSION CREATED")
print("="*80)
print("File: submission_neural_network.csv")
print(f"Predictions range: {test_predictions.min():.2f} to {test_predictions.max():.2f}")
print(f"Mean prediction: {test_predictions.mean():.2f}")

print("\n" + "="*80)
print("REALISTIC EXPECTATIONS")
print("="*80)
print("""
Neural Network Characteristics:
✓ Can capture non-linear patterns
✓ Flexible architecture
✓ Proven in many domains

BUT for this problem:
✗ Prone to CV overfitting (our enemy!)
✗ More complex than needed (simple > complex proven)
✗ Harder to regularize effectively
✗ Likely will follow the pattern: better CV → worse Kaggle

Prediction: This will score around 3.00-3.02 on Kaggle
           (despite potentially good CV score)

Why? Because we've proven 10 times that sophistication
     and CV optimization make Kaggle scores WORSE.

The simple blend at 2.98 remains the champion! 🏆
""")

print("\n✅ Done! Upload submission_neural_network.csv to Kaggle.")
print("🎯 Report back the score - let's see if the pattern holds!")
