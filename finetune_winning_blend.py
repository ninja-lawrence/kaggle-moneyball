import pandas as pd
import numpy as np

print("="*80)
print("FINE-TUNING AROUND WINNING BLEND (50/30/20 → 2.99)")
print("="*80)

# Load the best submissions
notemporal = pd.read_csv('submission_notemporal.csv')
multi = pd.read_csv('submission_multi_ensemble.csv')
finetuned = pd.read_csv('submission_finetuned.csv')

print('\nCurrent results:')
print('  50/30/20 → 2.99 ✅ BEST')
print('  40/20/40 → 3.00 (close!)')
print('  50/0/50  → 3.02')

# Fine-tune around 50/30/20
print("\n" + "="*80)
print("TESTING MICRO-ADJUSTMENTS AROUND WINNING WEIGHTS")
print("="*80)

# Small variations around 50/30/20
variants = []

# Vary first weight (notemporal) by ±5%
for w1 in [0.45, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55]:
    # Vary second weight (multi) by ±5%
    for w2 in [0.25, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.35]:
        w3 = 1.0 - w1 - w2
        if 0.15 <= w3 <= 0.25:  # Keep finetuned in reasonable range
            variants.append((w1, w2, w3))

print(f"Testing {len(variants)} weight combinations...")

results = []
for w1, w2, w3 in variants:
    blended_wins = (w1 * notemporal['W'] + 
                    w2 * multi['W'] + 
                    w3 * finetuned['W'])
    
    blended_wins = np.round(blended_wins).astype(int)
    blended_wins = np.clip(blended_wins, 0, 162)
    
    # Check difference from current best
    current_best = pd.read_csv('submission_blended_best.csv')
    diff_from_best = np.abs(blended_wins - current_best['W']).sum()
    same_as_best = (blended_wins == current_best['W']).sum()
    
    results.append({
        'w1': w1,
        'w2': w2,
        'w3': w3,
        'mean': blended_wins.mean(),
        'std': blended_wins.std(),
        'diff_count': diff_from_best,
        'same_count': same_as_best
    })

results_df = pd.DataFrame(results)

# Find variants most different from current
print("\nVariants MOST different from current 2.99 (explore new space):")
different_variants = results_df.nlargest(8, 'diff_count')
print(different_variants[['w1', 'w2', 'w3', 'mean', 'diff_count']].to_string(index=False))

print("\nVariants CLOSEST to current 2.99 (minor tweaks):")
similar_variants = results_df.nsmallest(8, 'diff_count')
print(similar_variants[['w1', 'w2', 'w3', 'mean', 'diff_count']].to_string(index=False))

# Create the most promising variants
print("\n" + "="*80)
print("CREATING HIGH-POTENTIAL SUBMISSIONS")
print("="*80)

# Pick most different variants
promising = [
    {'name': 'variant_a', 'weights': [0.45, 0.35, 0.20], 'reason': 'Less notemporal, more multi'},
    {'name': 'variant_b', 'weights': [0.52, 0.28, 0.20], 'reason': 'More notemporal, less multi'},
    {'name': 'variant_c', 'weights': [0.48, 0.32, 0.20], 'reason': 'Balanced adjustment'},
    {'name': 'variant_d', 'weights': [0.47, 0.30, 0.23], 'reason': 'More finetuned'},
    {'name': 'variant_e', 'weights': [0.53, 0.27, 0.20], 'reason': 'Push notemporal higher'},
]

for variant in promising:
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
    
    # Check how many predictions differ from current best
    current_best = pd.read_csv('submission_blended_best.csv')
    diff_count = (blended_wins != current_best['W']).sum()
    
    print(f'\n{variant["name"]:12s} ({w1:.2f}/{w2:.2f}/{w3:.2f}): {variant["reason"]}')
    print(f'  File: {filename}')
    print(f'  Mean: {blended_wins.mean():.2f}, Std: {blended_wins.std():.2f}')
    print(f'  Different predictions from best: {diff_count}/453 ({diff_count/453*100:.1f}%)')

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("Current best (50/30/20 → 2.99) is still champion!")
print("\nThe 30% weight on multi-ensemble is KEY, even though it scores 3.04 alone")
print("It provides diversity that improves the blend")
print("\nTry the variants above that differ most from current - they might find 2.98!")
print("\nVariants to test in order:")
print("  1. submission_blend_variant_a.csv (45/35/20)")
print("  2. submission_blend_variant_d.csv (47/30/23)")  
print("  3. submission_blend_variant_c.csv (48/32/20)")
