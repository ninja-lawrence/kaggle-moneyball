"""
═══════════════════════════════════════════════════════════════════════════════
🎯 FINAL MICRO-OPTIMIZATION: CHAMPION FINE-TUNING 🎯
═══════════════════════════════════════════════════════════════════════════════

Based on ALL previous attempts:
✅ Enhanced stacked: 2.97530 (TIED)
❌ Simple approaches: 3.06+ (WORSE)
❌ Complex approaches: 3.01+ (WORSE)

KEY INSIGHT: We can MATCH but not BEAT champion with meta-learning.

FINAL STRATEGY:
1. Recreate champion's 3 exact models
2. Micro-grid search AROUND 37/44/19 (±3% range)
3. Test with finer granularity (0.5% steps)
4. Look for 2.974 or 2.973 (tiny improvements)

This is the LAST attempt - if this fails, champion is optimal!

Date: October 7, 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 FINAL MICRO-OPTIMIZATION")
print("="*80)
print()
print("Strategy: Ultra-fine grid search around champion's 37/44/19")
print("Range: ±3% per weight, 0.5% granularity")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
y = train_df['W']

print(f"✓ Data loaded: Train={train_df.shape}, Test={test_df.shape}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# CHAMPION'S FEATURES (EXACT)
# ═══════════════════════════════════════════════════════════════════════════════

def create_champion_features(df):
    """Champion's exact feature set"""
    df = df.copy()
    
    if 'R' in df.columns and 'RA' in df.columns and 'G' in df.columns:
        for exp in [1.83, 1.85, 1.9, 2.0]:
            exp_str = str(int(exp * 100))
            df[f'pyth_exp_{exp_str}'] = df['R']**exp / (df['R']**exp + df['RA']**exp + 1)
            df[f'pyth_wins_{exp_str}'] = df[f'pyth_exp_{exp_str}'] * df['G']
        
        df['run_diff'] = df['R'] - df['RA']
        df['run_diff_per_game'] = df['run_diff'] / df['G']
        df['run_ratio'] = df['R'] / (df['RA'] + 1)
    
    if 'G' in df.columns:
        for col in ['R', 'RA', 'H', 'HR', 'BB', 'SO']:
            if col in df.columns:
                df[f'{col}_per_G'] = df[col] / df['G']
    
    if 'H' in df.columns and 'AB' in df.columns:
        df['BA'] = df['H'] / (df['AB'] + 1)
        if 'BB' in df.columns:
            df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
        if '2B' in df.columns and '3B' in df.columns and 'HR' in df.columns:
            singles = df['H'] - df['2B'] - df['3B'] - df['HR']
            df['SLG'] = (singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / (df['AB'] + 1)
            if 'OBP' in df.columns:
                df['OPS'] = df['OBP'] + df['SLG']
    
    if 'ERA' in df.columns and 'IPouts' in df.columns:
        if 'HA' in df.columns and 'BBA' in df.columns:
            df['WHIP'] = (df['HA'] + df['BBA']) / ((df['IPouts'] / 3) + 1)
        if 'SOA' in df.columns:
            df['K_per_9'] = (df['SOA'] * 27) / (df['IPouts'] + 1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE CHAMPION'S 3 MODELS (EXACT RECREATION)
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🏆 RECREATING CHAMPION'S 3 MODELS")
print("="*80)

train_feat = create_champion_features(train_df.copy())
test_feat = create_champion_features(test_df.copy())

exclude_cols = {
    'W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'mlb_rpg'
}

feature_cols = sorted(list((set(train_feat.columns) & set(test_feat.columns)) - exclude_cols))
X_train = train_feat[feature_cols].fillna(0)
X_test = test_feat[feature_cols].fillna(0)

print(f"✓ Features: {len(feature_cols)}")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Ridge alpha=10
print("\n[1/3] Model 1: Ridge α=10 (Notemporal)")
model1 = Ridge(alpha=10.0, random_state=42)
model1.fit(X_train_scaled, y)
pred1 = model1.predict(X_test_scaled)

# Model 2: Ridge alpha=3
print("[2/3] Model 2: Ridge α=3 (Multi-ensemble)")
model2 = Ridge(alpha=3.0, random_state=42)
model2.fit(X_train_scaled, y)
pred2 = model2.predict(X_test_scaled)

# Model 3: Ridge alpha=10, multi-seed
print("[3/3] Model 3: Ridge α=10, multi-seed (Finetuned)")
seeds = [42, 123, 456]
preds_multi = []
for seed in seeds:
    model = Ridge(alpha=10.0, random_state=seed)
    model.fit(X_train_scaled, y)
    preds_multi.append(model.predict(X_test_scaled))
pred3 = np.mean(preds_multi, axis=0)

print("\n✓ All 3 champion models recreated")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE OUT-OF-FOLD PREDICTIONS FOR OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🔍 GENERATING OOF PREDICTIONS (for weight optimization)")
print("="*80)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train), 3))

print("Generating OOF predictions...")
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    # Model 1
    m1 = Ridge(alpha=10.0, random_state=42)
    m1.fit(X_train_scaled[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 0] = m1.predict(X_train_scaled[val_idx])
    
    # Model 2
    m2 = Ridge(alpha=3.0, random_state=42)
    m2.fit(X_train_scaled[train_idx], y.iloc[train_idx])
    oof_preds[val_idx, 1] = m2.predict(X_train_scaled[val_idx])
    
    # Model 3 (multi-seed)
    fold_preds = []
    for seed in [42, 123, 456]:
        m3 = Ridge(alpha=10.0, random_state=seed)
        m3.fit(X_train_scaled[train_idx], y.iloc[train_idx])
        fold_preds.append(m3.predict(X_train_scaled[val_idx]))
    oof_preds[val_idx, 2] = np.mean(fold_preds, axis=0)

print("✓ OOF predictions ready")

# ═══════════════════════════════════════════════════════════════════════════════
# ULTRA-FINE GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🔬 ULTRA-FINE GRID SEARCH")
print("="*80)

# Champion weights: 37/44/19
# Search: ±3% range, 0.5% granularity

print("\nSearching around champion 37/44/19...")
print("Range: [34-40] / [41-47] / [16-22] %")
print("Granularity: 0.5%")
print()

best_weights = None
best_mae = float('inf')
all_results = []

# Generate weight combinations
w1_range = np.arange(0.34, 0.41, 0.005)  # 34% to 40%
w2_range = np.arange(0.41, 0.48, 0.005)  # 41% to 47%

total_tests = 0
for w1 in w1_range:
    for w2 in w2_range:
        w3 = 1.0 - w1 - w2
        
        # Constraint: w3 must be in [0.16, 0.22] range
        if w3 < 0.16 or w3 > 0.22:
            continue
        
        # Calculate OOF MAE
        oof_blend = w1 * oof_preds[:, 0] + w2 * oof_preds[:, 1] + w3 * oof_preds[:, 2]
        mae = mean_absolute_error(y, oof_blend)
        
        all_results.append((w1, w2, w3, mae))
        total_tests += 1
        
        if mae < best_mae:
            best_mae = mae
            best_weights = (w1, w2, w3)

print(f"✓ Tested {total_tests} weight combinations")
print()

# Sort results and show top 10
all_results_sorted = sorted(all_results, key=lambda x: x[3])

print("🏆 TOP 10 WEIGHT COMBINATIONS:")
print("-" * 80)
print(f"{'Rank':<6} {'W1 (%)':>8} {'W2 (%)':>8} {'W3 (%)':>8} {'OOF MAE':>10} {'vs Best':>10}")
print("-" * 80)

for i, (w1, w2, w3, mae) in enumerate(all_results_sorted[:10], 1):
    diff = mae - best_mae
    diff_str = f"+{diff:.5f}" if diff > 0 else "BEST"
    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{marker} {i:<3} {w1*100:>8.1f} {w2*100:>8.1f} {w3*100:>8.1f} {mae:>10.5f} {diff_str:>10}")

print("-" * 80)

# Champion baseline for comparison
champion_w1, champion_w2, champion_w3 = 0.37, 0.44, 0.19
champion_oof = champion_w1 * oof_preds[:, 0] + champion_w2 * oof_preds[:, 1] + champion_w3 * oof_preds[:, 2]
champion_mae = mean_absolute_error(y, champion_oof)

print(f"\n📊 Champion (37/44/19): {champion_mae:.5f} OOF MAE")
print(f"🏆 Best Found: {best_mae:.5f} OOF MAE")

if best_mae < champion_mae:
    improvement = ((champion_mae - best_mae) / champion_mae) * 100
    print(f"✅ Improvement: {improvement:.3f}% better!")
else:
    print(f"⚖️  Champion is optimal (within tested range)")

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE SUBMISSIONS FOR TOP 5
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("💾 SAVING TOP 5 SUBMISSIONS")
print("="*80)

for rank, (w1, w2, w3, mae) in enumerate(all_results_sorted[:5], 1):
    # Create blended predictions
    pred_blend = w1 * pred1 + w2 * pred2 + w3 * pred3
    pred_blend = np.clip(pred_blend, 0, 162).round().astype(int)
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'W': pred_blend
    })
    
    filename = f'submission_micro_rank{rank}_w{int(w1*100)}_{int(w2*100)}_{int(w3*100)}.csv'
    submission.to_csv(filename, index=False)
    
    print(f"✓ Rank {rank}: {filename}")
    print(f"  Weights: {w1*100:.1f}% / {w2*100:.1f}% / {w3*100:.1f}%  (OOF: {mae:.5f})")

print()
print("Sample predictions (best model):")
best_w1, best_w2, best_w3, _ = all_results_sorted[0]
best_pred = best_w1 * pred1 + best_w2 * pred2 + best_w3 * pred3
best_pred_final = np.clip(best_pred, 0, 162).round().astype(int)

best_sub = pd.DataFrame({
    'ID': test_df['ID'],
    'W': best_pred_final
})
print(best_sub.head(10))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("🎉 MICRO-OPTIMIZATION COMPLETE!")
print("="*80)
print()
print(f"📊 Search Statistics:")
print(f"  • Weight combinations tested: {total_tests}")
print(f"  • Search space: ±3% around champion")
print(f"  • Granularity: 0.5% steps")
print()
print(f"🏆 Best Configuration:")
print(f"  • Weights: {best_weights[0]*100:.1f}% / {best_weights[1]*100:.1f}% / {best_weights[2]*100:.1f}%")
print(f"  • OOF MAE: {best_mae:.5f}")
print()
print(f"📊 Champion Baseline:")
print(f"  • Weights: 37.0% / 44.0% / 19.0%")
print(f"  • OOF MAE: {champion_mae:.5f}")
print()

if best_mae < champion_mae:
    print("✅ POTENTIAL IMPROVEMENT FOUND!")
    print(f"   OOF improvement: {((champion_mae - best_mae) / champion_mae) * 100:.3f}%")
    print()
    print("🎯 Next Step: Submit top ranked file to Kaggle")
    print("   Expected: Similar or slightly better than 2.97530")
else:
    print("⚖️  CHAMPION CONFIRMED OPTIMAL")
    print("   No better weights found in ±3% range")
    print("   This validates the original 37/44/19 blend!")
print()
print("📁 Submissions created:")
for i in range(1, 6):
    w1, w2, w3, _ = all_results_sorted[i-1]
    filename = f'submission_micro_rank{i}_w{int(w1*100)}_{int(w2*100)}_{int(w3*100)}.csv'
    print(f"  {i}. {filename}")
print()
print("🚀 Ready to submit!")
print()
print("="*80)
