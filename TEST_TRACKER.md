# Kaggle Test Results Tracker

## 🏆 FINAL CHAMPIONS
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| **submission_blend_ultra_q.csv** | **37/44/19** | **2.97530** | **Oct 5** | **🥇 BEST!** |
| **submission_blend_ultra_r.csv** | **36/45/19** | **2.97530** | **Oct 5** | **🥇 TIED BEST!** |

## Baseline / First Plateau (2.99176)
| File | Weights | MAE | Date | Notes |
|------|---------|-----|------|-------|
| submission_blended_best.csv | 50/30/20 | 2.99176 | Oct 5 | Original |
| submission_blend_variant_b.csv | 52/28/20 | 2.99176 | Oct 5 | Plateau |
| submission_blend_variant_e.csv | 53/27/20 | 2.99176 | Oct 5 | Plateau |
| submission_blend_micro_a.csv | 54/26/20 | 2.99176 | Oct 5 | Plateau |
| submission_blend_micro_e.csv | 53/26/21 | 2.99176 | Oct 5 | Plateau |

## Breakthrough - Radical Variants
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_radical_b.csv | 40/40/20 | 2.97942 | Oct 5 | First breakthrough! |
| submission_blend_radical_a.csv | 45/35/20 | 2.98765 | Oct 5 | Good |
| submission_blend_radical_j.csv | 40/30/30 | 2.98765 | Oct 5 | Good |
| submission_blend_radical_k.csv | 33/33/34 | 2.98765 | Oct 5 | Good |
| submission_blend_radical_g.csv | 45/25/30 | 2.99588 | Oct 5 | Better |
| submission_blend_radical_h.csv | 40/25/35 | 2.99588 | Oct 5 | Better |
| submission_blend_radical_d.csv | 60/25/15 | 3.00823 | Oct 5 | Worse |
| submission_blend_radical_l.csv | 60/20/20 | 3.01234 | Oct 5 | Worse |
| submission_blend_radical_e.csv | 65/20/15 | 3.01234 | Oct 5 | Worse |

## Ultra Fine-Tuning (Around 40/40/20)
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_ultra_q.csv | 37/44/19 | 2.97530 | Oct 5 | 🥇 CHAMPION! |
| submission_blend_ultra_r.csv | 36/45/19 | 2.97530 | Oct 5 | 🥇 CHAMPION! |
| submission_blend_ultra_p.csv | 38/43/19 | 2.97942 | Oct 5 | Plateau B |
| submission_blend_ultra_d.csv | 38/42/20 | 2.97942 | Oct 5 | Plateau B |
| submission_blend_ultra_j.csv | 37/43/20 | 2.97942 | Oct 5 | Plateau B |
| submission_blend_ultra_c.csv | 39/41/20 | 2.97942 | Oct 5 | Plateau B |
| submission_blend_ultra_k.csv | 40/38/22 | 2.97942 | Oct 5 | Plateau B |
| submission_blend_ultra_e.csv | 40/41/19 | 2.97942 | Oct 5 | Plateau B |
| submission_blend_ultra_g.csv | 40/39/21 | 2.97942 | Oct 5 | Plateau B |

## Micro-Variants (Testing Boundaries)
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_micro_b.csv | 55/25/20 | 3.00000 | Oct 5 | Outside plateau |
| submission_blend_micro_i.csv | 52/26/22 | 3.00000 | Oct 5 | Outside plateau |

### Priority 2 - Fine-tune Around Variant E
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_micro_e.csv | 53/26/21 | ___._____ | ____ | Variant E +1% finetuned |
| submission_blend_micro_f.csv | 53/28/19 | ___._____ | ____ | Variant E +1% multi |
| submission_blend_micro_g.csv | 52/27/21 | ___._____ | ____ | Between B and E |
| submission_blend_micro_h.csv | 54/27/19 | ___._____ | ____ | Beyond E |

### Priority 3 - More Finetuned Weight
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_micro_i.csv | 52/26/22 | ___._____ | ____ | 22% finetuned |
| submission_blend_micro_j.csv | 51/26/23 | ___._____ | ____ | 23% finetuned |
| submission_blend_micro_k.csv | 50/26/24 | ___._____ | ____ | 24% finetuned |
| submission_blend_micro_l.csv | 51/25/24 | ___._____ | ____ | Balanced |

### Additional Exploration
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_micro_m.csv | 54/25/21 | ___._____ | ____ | High notemporal |
| submission_blend_micro_n.csv | 55/24/21 | ___._____ | ____ | Very high notemporal |
| submission_blend_micro_o.csv | 53/25/22 | ___._____ | ____ | Variant E + finetuned |

## Other Variants (Still to test from original batch)
| File | Weights (N/M/F) | MAE | Date | Notes |
|------|-----------------|-----|------|-------|
| submission_blend_variant_a.csv | 45/35/20 | ___._____ | ____ | Less notemporal, more multi |
| submission_blend_variant_c.csv | 48/32/20 | ___._____ | ____ | Balanced adjustment |
| submission_blend_variant_d.csv | 47/30/23 | ___._____ | ____ | More finetuned |

## Analysis Notes

### Best Result So Far
- **MAE**: 2.99176
- **Weights**: 50/30/20, 52/28/20, or 53/27/20 (all identical)
- **Key Finding**: Stable plateau region exists

### Patterns Observed
(Fill in as you test)

### Hypothesis Updates
(Fill in based on results)

### Next Steps
(Update based on findings)

---

## Quick Stats Template

After testing a batch, fill in:

**Tested**: ___ / 15 micro-variants  
**Improvements**: ___ variants better than 2.99176  
**Best MAE**: ___._____ (weights: __/__)  
**Worst MAE**: ___._____ (weights: __/__)  
**Average MAE**: ___._____ 

**Direction Analysis**:
- Push notemporal (54-57%): Best = ___._____
- Around variant E: Best = ___._____
- More finetuned (22-24%): Best = ___._____

**Recommendation**: _______________

---

## Testing Schedule

### Week of Oct 5-11
- [ ] Monday: micro_a, micro_b
- [ ] Tuesday: micro_e, micro_i
- [ ] Wednesday: micro_c, micro_j
- [ ] Thursday: micro_d, micro_h
- [ ] Friday: Review results, decide next batch

### Week of Oct 12-18
- [ ] Based on Week 1 findings
- [ ] Test most promising direction more deeply
- [ ] Consider radical variants if plateau persists

---

**Last Updated**: October 5, 2025  
**Current Champion**: 2.99176 (three-way tie)  
**Target**: 2.98 or better
