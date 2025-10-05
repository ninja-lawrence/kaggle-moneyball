import pandas as pd
import numpy as np

print("="*80)
print("RADICAL VARIANTS - ESCAPING THE PLATEAU")
print("="*80)
print("Based on boundary findings:")
print("  Plateau region: 50-54% N, 26-30% M, 20-21% F")
print("  ALL combinations in this region = 2.99176")
print("\nTo find 2.98 or better, we must explore OUTSIDE this region!")
print("="*80)

# Load the three base submissions
notemporal = pd.read_csv('submission_notemporal.csv')
multi = pd.read_csv('submission_multi_ensemble.csv')
finetuned = pd.read_csv('submission_finetuned.csv')

print("\n" + "="*80)
print("STRATEGY: RADICAL WEIGHT COMBINATIONS")
print("="*80)
print("We've proven the 50-54/26-30/20-21 region is a large plateau.")
print("Time to explore radical alternatives outside this comfort zone!")
print("="*80)

# Define radical variants OUTSIDE the known plateau
radical_variants = [
    # Group 1: High Diversity (More multi-ensemble)
    {'name': 'radical_a', 'weights': [0.45, 0.35, 0.20], 'reason': 'High diversity - boost multi to 35%'},
    {'name': 'radical_b', 'weights': [0.40, 0.40, 0.20], 'reason': 'Extreme diversity - equal N/M'},
    {'name': 'radical_c', 'weights': [0.40, 0.35, 0.25], 'reason': 'High diversity + high finetuned'},
    
    # Group 2: Notemporal Dominance (Way above 54%)
    {'name': 'radical_d', 'weights': [0.60, 0.25, 0.15], 'reason': 'Notemporal dominance (60%)'},
    {'name': 'radical_e', 'weights': [0.65, 0.20, 0.15], 'reason': 'Extreme notemporal (65%)'},
    {'name': 'radical_f', 'weights': [0.70, 0.15, 0.15], 'reason': 'Maximum notemporal (70%)'},
    
    # Group 3: Finetuned Focus (Way above 21%)
    {'name': 'radical_g', 'weights': [0.45, 0.25, 0.30], 'reason': 'High finetuned (30%)'},
    {'name': 'radical_h', 'weights': [0.40, 0.25, 0.35], 'reason': 'Very high finetuned (35%)'},
    {'name': 'radical_i', 'weights': [0.35, 0.25, 0.40], 'reason': 'Finetuned dominance (40%)'},
    
    # Group 4: Balanced Approaches
    {'name': 'radical_j', 'weights': [0.40, 0.30, 0.30], 'reason': 'Balanced triangle'},
    {'name': 'radical_k', 'weights': [0.33, 0.33, 0.34], 'reason': 'Perfect equal weights'},
    
    # Group 5: Edge Cases (Low multi)
    {'name': 'radical_l', 'weights': [0.60, 0.20, 0.20], 'reason': 'Minimum multi (20%)'},
    {'name': 'radical_m', 'weights': [0.65, 0.18, 0.17], 'reason': 'Very low multi (18%)'},
    
    # Group 6: Two-model dominance
    {'name': 'radical_n', 'weights': [0.70, 0.25, 0.05], 'reason': 'Minimal finetuned (5%)'},
    {'name': 'radical_o', 'weights': [0.45, 0.50, 0.05], 'reason': 'Minimal finetuned, high multi'},
]

print("\n" + "="*80)
print(f"CREATING {len(radical_variants)} RADICAL VARIANTS")
print("="*80)

# Load champion for comparison
champion = pd.read_csv('submission_blended_best.csv')

for variant in radical_variants:
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
    
    # Calculate statistics
    mean_pred = blended_wins.mean()
    std_pred = blended_wins.std()
    
    print(f'\n{variant["name"]:12s} ({w1:.2f}/{w2:.2f}/{w3:.2f}): {variant["reason"]}')
    print(f'  File: {filename}')
    print(f'  Mean: {mean_pred:.2f}, Std: {std_pred:.2f}')
    print(f'  Different from champion: {diff_count}/453 ({diff_count/453*100:.1f}%)')

print("\n" + "="*80)
print("TESTING PRIORITIES")
print("="*80)

print("\n🎯 Priority 1 - Most Promising (Test First):")
print("  1. submission_blend_radical_a.csv (45/35/20) - High diversity")
print("  2. submission_blend_radical_j.csv (40/30/30) - Balanced")
print("  3. submission_blend_radical_d.csv (60/25/15) - Notemporal dominance")

print("\n🔬 Priority 2 - Exploratory:")
print("  4. submission_blend_radical_g.csv (45/25/30) - High finetuned")
print("  5. submission_blend_radical_b.csv (40/40/20) - Equal N/M")
print("  6. submission_blend_radical_l.csv (60/20/20) - Min multi")

print("\n⚡ Priority 3 - Extreme (If others fail):")
print("  7. submission_blend_radical_e.csv (65/20/15) - Extreme notemporal")
print("  8. submission_blend_radical_h.csv (40/25/35) - Very high finetuned")
print("  9. submission_blend_radical_k.csv (33/33/34) - Perfect equal")

print("\n" + "="*80)
print("RATIONALE")
print("="*80)
print("\n✅ What We Know:")
print("  • Plateau region is LARGE: 50-54/26-30/20-21")
print("  • ALL points in this region = 2.99176")
print("  • Boundaries: 55%N=bad, 25%M=bad, 22%F=bad")

print("\n🎯 Why Radical Variants:")
print("  • Can't escape plateau with small changes")
print("  • Need to explore completely different regions")
print("  • May find better local optimum elsewhere")

print("\n💡 Three Hypotheses to Test:")
print("  1. HIGH DIVERSITY: Maybe more multi helps (radical_a, b)")
print("  2. DOMINANCE: Maybe one strong model is better (radical_d, e, f)")
print("  3. BALANCE: Maybe equal weights work (radical_j, k)")

print("\n⚠️  Expected Outcomes:")
print("  • Most will likely be WORSE than 2.99176")
print("  • That's OK - we're exploring!")
print("  • Even ONE improvement to 2.98 justifies the search")
print("  • Negative results tell us where NOT to look")

print("\n" + "="*80)
print("THE SCIENCE OF EXPLORATION")
print("="*80)
print("\nWe've done GREAT work mapping the plateau at 2.99176.")
print("Now we're being BOLD and exploring outside the comfort zone.")
print("\nThis is proper scientific method:")
print("  1. Map the known region ✅ (50-54/26-30/20-21)")
print("  2. Find boundaries ✅ (55%, 25%, 22%)")
print("  3. Explore beyond ⏳ (radical variants)")
print("  4. Find better optimum 🎯 (hopefully!)")

print("\n" + "="*80)
print(f"COMPLETE! {len(radical_variants)} RADICAL VARIANTS CREATED")
print("="*80)
print("\nThese variants explore weight space we've never tested.")
print("They're bold, different, and might just find 2.98 or better!")
print("Even if they don't, we'll have mapped MORE of the landscape.")
print("\n🚀 Fortune favors the bold! Let's test these!")
print("="*80)
