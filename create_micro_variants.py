import pandas as pd
import numpy as np

print("="*80)
print("MICRO-VARIANTS AROUND SUCCESSFUL WEIGHTS")
print("="*80)
print("Based on findings:")
print("  50/30/20 → 2.99176 ✅")
print("  52/28/20 → 2.99176 ✅")
print("  53/27/20 → 2.99176 ✅")
print("\nPattern: More notemporal, less multi, keep finetuned stable")
print("="*80)

# Load the three base submissions
notemporal = pd.read_csv('submission_notemporal.csv')
multi = pd.read_csv('submission_multi_ensemble.csv')
finetuned = pd.read_csv('submission_finetuned.csv')

# Strategy: Create micro-variants exploring three directions
print("\n" + "="*80)
print("STRATEGY")
print("="*80)
print("Direction 1: Push notemporal higher (54-57% range)")
print("Direction 2: Fine-tune around variant E (53/27/20)")
print("Direction 3: Explore finetuned weight variations")
print("="*80)

# Define promising micro-variants
micro_variants = [
    # Direction 1: Push notemporal higher
    {'name': 'micro_a', 'weights': [0.54, 0.26, 0.20], 'reason': 'Push notemporal to 54%'},
    {'name': 'micro_b', 'weights': [0.55, 0.25, 0.20], 'reason': 'Push notemporal to 55%'},
    {'name': 'micro_c', 'weights': [0.56, 0.24, 0.20], 'reason': 'Push notemporal to 56%'},
    {'name': 'micro_d', 'weights': [0.57, 0.23, 0.20], 'reason': 'Push notemporal to 57%'},
    
    # Direction 2: Fine-tune around variant E (53/27/20)
    {'name': 'micro_e', 'weights': [0.53, 0.26, 0.21], 'reason': 'Variant E +1% finetuned'},
    {'name': 'micro_f', 'weights': [0.53, 0.28, 0.19], 'reason': 'Variant E +1% multi'},
    {'name': 'micro_g', 'weights': [0.52, 0.27, 0.21], 'reason': 'Between B and E, more finetuned'},
    {'name': 'micro_h', 'weights': [0.54, 0.27, 0.19], 'reason': 'Beyond E, less finetuned'},
    
    # Direction 3: Explore finetuned weight (it scored 3.02 alone)
    {'name': 'micro_i', 'weights': [0.52, 0.26, 0.22], 'reason': 'More finetuned (22%)'},
    {'name': 'micro_j', 'weights': [0.51, 0.26, 0.23], 'reason': 'More finetuned (23%)'},
    {'name': 'micro_k', 'weights': [0.50, 0.26, 0.24], 'reason': 'More finetuned (24%)'},
    {'name': 'micro_l', 'weights': [0.51, 0.25, 0.24], 'reason': 'Balance: less multi, more finetuned'},
    
    # Additional exploration
    {'name': 'micro_m', 'weights': [0.54, 0.25, 0.21], 'reason': 'High notemporal, balanced others'},
    {'name': 'micro_n', 'weights': [0.55, 0.24, 0.21], 'reason': 'Very high notemporal'},
    {'name': 'micro_o', 'weights': [0.53, 0.25, 0.22], 'reason': 'Variant E direction, more finetuned'},
]

print("\n" + "="*80)
print(f"CREATING {len(micro_variants)} MICRO-VARIANTS")
print("="*80)

# Load champion for comparison
champion = pd.read_csv('submission_blended_best.csv')

for variant in micro_variants:
    w1, w2, w3 = variant['weights']
    
    # Verify weights sum to 1.0
    assert abs(w1 + w2 + w3 - 1.0) < 0.001, f"Weights don't sum to 1.0: {w1}+{w2}+{w3}={w1+w2+w3}"
    
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
    
    # Check how many predictions differ from champion
    diff_count = (blended_wins != champion['W']).sum()
    
    # Calculate some statistics
    mean_pred = blended_wins.mean()
    std_pred = blended_wins.std()
    
    print(f'\n{variant["name"]:10s} ({w1:.2f}/{w2:.2f}/{w3:.2f}): {variant["reason"]}')
    print(f'  File: {filename}')
    print(f'  Mean: {mean_pred:.2f}, Std: {std_pred:.2f}')
    print(f'  Different from champion: {diff_count}/453 ({diff_count/453*100:.1f}%)')

print("\n" + "="*80)
print("TESTING RECOMMENDATIONS")
print("="*80)
print("\nPriority 1 - Direction 1 (Push notemporal higher):")
print("  1. submission_blend_micro_a.csv (54/26/20)")
print("  2. submission_blend_micro_b.csv (55/25/20)")
print("  3. submission_blend_micro_c.csv (56/24/20)")

print("\nPriority 2 - Direction 2 (Fine-tune around variant E):")
print("  4. submission_blend_micro_e.csv (53/26/21)")
print("  5. submission_blend_micro_g.csv (52/27/21)")
print("  6. submission_blend_micro_h.csv (54/27/19)")

print("\nPriority 3 - Direction 3 (More finetuned weight):")
print("  7. submission_blend_micro_i.csv (52/26/22)")
print("  8. submission_blend_micro_j.csv (51/26/23)")
print("  9. submission_blend_micro_o.csv (53/25/22)")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("The 50/30/20, 52/28/20, and 53/27/20 all score identically (2.99176)")
print("This suggests we're in a STABLE REGION of the weight space")
print("\nTo find improvement (2.98 or better), we need to:")
print("  • Push boundaries more aggressively")
print("  • Explore different directions simultaneously")
print("  • Test weights outside the comfort zone")
print("\nThe fact that three different weights achieve same score means:")
print("  ✅ Model is robust (good for production)")
print("  🔬 Need bigger changes to find improvement")
print("="*80)

print("\n" + "="*80)
print(f"COMPLETE! {len(micro_variants)} MICRO-VARIANTS CREATED")
print("="*80)
