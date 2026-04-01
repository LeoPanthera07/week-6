# Extra Trees vs Random Forest – Comparison

## Splitting Strategy
- Random Forest selects the best split among a random subset of features at each node.
- Extra Trees randomly selects thresholds for each candidate feature and picks the best among those random thresholds.
- This additional randomness reduces sensitivity to noise and typically speeds up training.

## Speed – Actual Results on Loan Dataset
| Model | Training Time |
|---|---|
| Random Forest (tuned) | 0.448 seconds |
| Extra Trees (n_estimators=300) | 0.659 seconds |

On this dataset Extra Trees was slightly slower.
This can happen when n_estimators is higher for Extra Trees (300) vs the tuned Random Forest (200).
When matched on n_estimators, Extra Trees is usually faster because it skips the best-split search.

## Performance – Actual Results on Loan Dataset
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.9483 | 0.9512 | 0.9907 |
| Extra Trees | 0.9567 | 0.9592 | 0.9940 |

Extra Trees outperformed Random Forest slightly on all three metrics.
This is consistent with the known behavior: extreme randomness can act as useful regularization on clean datasets.

## Practical Takeaway
- Use Random Forest when you want stable, well-tuned performance.
- Use Extra Trees when you want comparable or better performance with simpler setup, especially on clean structured data.
- On this synthetic loan dataset, Extra Trees gives better accuracy and ROC-AUC, making it a strong alternative.
