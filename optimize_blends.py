import pandas as pd
import numpy as np
from itertools import product

print("="*80)
print("OPTIMIZING BLEND WEIGHTS")
print("="*80)

# Load the best submissions
notemporal = pd.read_csv('submission_notemporal.csv')
multi = pd.read_csv('submission_multi_ensemble.csv')
finetuned = pd.read_csv('submission_finetuned.csv')

print('\nLoaded submissions:')
print(f'  notemporal:     Kaggle 3.03')
print(f'  multi_ensemble: Kaggle 3.04')
print(f'  finetuned:      Kaggle 3.02')
print(f'  Current best blend: Kaggle 2.99 (50/30/20)')

# Try different weight combinations
print("\n" + "="*80)
print("TESTING WEIGHT COMBINATIONS")
print("="*80)

# Generate weight combinations that sum to 1.0
weight_combos = []
for w1 in range(2, 9):  # 20% to 80%
    for w2 in range(1, 9):
        for w3 in range(1, 9):
            if w1 + w2 + w3 == 10:  # Sum to 100%
                weight_combos.append((w1/10, w2/10, w3/10))

results = []

for w1, w2, w3 in weight_combos:
    blended_wins = (w1 * notemporal['W'] + 
                    w2 * multi['W'] + 
                    w3 * finetuned['W'])
    
    blended_wins = np.round(blended_wins).astype(int)
    blended_wins = np.clip(blended_wins, 0, 162)
    
    # Calculate statistics
    mean_diff = abs(blended_wins.mean() - 79.0)  # Close to 79 wins average
    std = blended_wins.std()
    
    # Check how different it is from current best
    current_best = pd.read_csv('submission_blended_best.csv')
    diff_from_best = np.abs(blended_wins - current_best['W']).mean()
    
    results.append({
        'w_notemporal': w1,
        'w_multi': w2,
        'w_finetuned': w3,
        'mean': blended_wins.mean(),
        'std': std,
        'diff_from_current': diff_from_best
    })

# Sort by different from current (to find variations)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('diff_from_current')

print("\nTop 10 blends (most different from current best 2.99):")
print(results_df.tail(10)[['w_notemporal', 'w_multi', 'w_finetuned', 'mean', 'std', 'diff_from_current']].to_string(index=False))

print("\nTop 10 blends (most similar to current best 2.99):")
print(results_df.head(10)[['w_notemporal', 'w_multi', 'w_finetuned', 'mean', 'std', 'diff_from_current']].to_string(index=False))

# Create a few interesting variants
variants = [
    # More weight on best individual model (finetuned 3.02)
    {'name': 'finetuned_heavy', 'weights': [0.4, 0.2, 0.4]},
    # Equal weights
    {'name': 'equal', 'weights': [0.33, 0.34, 0.33]},
    # Focus on top 2
    {'name': 'top2_only', 'weights': [0.5, 0.0, 0.5]},
    # Current best tweaked
    {'name': 'tweak_v1', 'weights': [0.45, 0.35, 0.2]},
    {'name': 'tweak_v2', 'weights': [0.55, 0.25, 0.2]},
    # More conservative (higher weight on proven 3.03)
    {'name': 'conservative', 'weights': [0.6, 0.2, 0.2]},
]

print("\n" + "="*80)
print("CREATING VARIANT SUBMISSIONS")
print("="*80)

for variant in variants:
    w1, w2, w3 = variant['weights']
    
    blended_wins = (w1 * notemporal['W'] + 
                    w2 * multi['W'] + 
                    w3 * finetuned['W'])
    
    blended_wins = np.round(blended_wins).astype(int)
    blended_wins = np.clip(blended_wins, 0, 162)
    
    submission_df = pd.DataFrame({
        'ID': notemporal['ID'],
        'W': blended_wins
    })
    
    filename = f'submission_blend_{variant["name"]}.csv'
    submission_df.to_csv(filename, index=False)
    
    print(f'\n{variant["name"]:15s} ({w1:.0%}/{w2:.0%}/{w3:.0%}):')
    print(f'  File: {filename}')
    print(f'  Mean: {blended_wins.mean():.2f}, Std: {blended_wins.std():.2f}')
    print(f'  Range: {blended_wins.min()}-{blended_wins.max()}')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Current best: submission_blended_best.csv (50/30/20) → 2.99")
print("\nNew variants created for testing:")
print("  1. submission_blend_top2_only.csv - Only best 2 models")
print("  2. submission_blend_finetuned_heavy.csv - More weight on finetuned")
print("  3. submission_blend_equal.csv - Equal weights")
print("  4. submission_blend_tweak_v1.csv - Slight adjustment")
print("  5. submission_blend_tweak_v2.csv - Slight adjustment")
print("\nStrategy: Try variants that are most different from current 2.99")
print("This explores different parts of the solution space")
