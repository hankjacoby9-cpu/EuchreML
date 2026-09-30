# Opponent-pool final evaluation

The candidate is `checkpoints/card_play_pool_best.npz`, trained for ten
generations from `card_play_match_best.npz`. Training used the heuristic,
fixed bidding policy, `card_play_best.npz`, the starting champion, and promoted
generation champions.

The locked test uses seeds 9,000,001 through 9,000,500. These 500 seeds are
outside the training range beginning at 1 and validation range beginning at
1,000,001. Each seed is evaluated from all four starting dealers with the
candidate assigned to both partnerships, producing 4,000 matches per opponent.
Confidence intervals use 1,000 bootstrap samples over the paired partnership
results for each seed and dealer.

| Opponent | Win rate (95% CI) | Capped margin (95% CI) |
| --- | --- | --- |
| Heuristic | 0.8902 [0.8808, 0.8995] | +4.6848 [+4.5735, +4.7852] |
| Previous champion | 0.5333 [0.5225, 0.5450] | +0.4098 [+0.3277, +0.4893] |

Calling diagnostics use one-hand head-to-head replays on the same locked deals.
Calling success means the candidate partnership won at least three tricks after
calling trump. Euchre rate is the share of its calls that won fewer than three
tricks. Lone success uses the same three-trick threshold; a lone sweep requires
all five tricks.

| Opponent | Calls | Call success | Euchre rate | Lone calls | Lone success | Lone sweeps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Heuristic | 835 | 0.9269 | 0.0731 | 166 | 0.9759 | 32 (0.1928) |
| Previous champion | 2,000 | 0.8590 | 0.1410 | 406 | 0.9187 | 36 (0.0887) |

Reproduce the report with:

```sh
.venv/bin/python scripts/final_evaluation.py \
    checkpoints/card_play_pool_best.npz \
    checkpoints/card_play_match_best.npz \
    --seeds 500 --bootstrap-samples 1000
```
