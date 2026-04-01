# Day 32 – Part D: AI-Augmented Visualization

## Prompt Used
`Explain Decision Tree vs Random Forest vs Logistic Regression with a simple matplotlib infographic for non-technical audience.`

## AI Output Structure (Generated and then corrected)

Three columns, one per model.
Each column shows:
- When to use
- Accuracy (from our actual results)
- Interpretability
- Speed

## Infographic Code (Corrected and Improved)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.suptitle("Model Comparison: Decision Tree vs Random Forest vs Logistic Regression", fontsize=13, y=1.02)

models = ["Decision Tree", "Random Forest", "Logistic Regression"]
colors = ["#4C72B0", "#55A868", "#C44E52"]
interpret = [3, 1.5, 3]
accuracy = [2, 3, 1.5]

when_to_use = [
    "Need to explain decisions\nto regulators or clients",
    "Best accuracy matters\nand speed is acceptable",
    "Simple baseline model\nor linear separation"
]
pros = [
    "Easy to explain\nDecision rules visible",
    "High accuracy\nRobust to noise",
    "Fast training\nSimple and stable"
]
cons = [
    "Overfits easily\nSensitive to data changes",
    "Hard to fully explain\nSlower to train",
    "Cannot model\nnon-linear patterns"
]
results = [
    "Acc: 0.8767\nROC-AUC: 0.9555",
    "Acc: 0.9483\nROC-AUC: 0.9907",
    "Acc: ~0.85 (baseline)"
]

for ax, name, color, wtu, p, c, r in zip(axes, models, colors, when_to_use, pros, cons, results):
    ax.set_facecolor(color + "22")
    ax.axis("off")
    ax.text(0.5, 0.92, name, ha="center", va="top", fontsize=12, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.78, f"When to use:
{wtu}", ha="center", va="top", fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.55, f"Pros:
{p}", ha="center", va="top", fontsize=9, color="#2d6a2d", transform=ax.transAxes)
    ax.text(0.5, 0.33, f"Cons:
{c}", ha="center", va="top", fontsize=9, color="#a00000", transform=ax.transAxes)
    ax.text(0.5, 0.12, f"Loan Dataset:
{r}", ha="center", va="top", fontsize=9, color="#333333", transform=ax.transAxes)
    rect = mpatches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02", linewidth=2, edgecolor=color, facecolor="white", transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig("model_comparison_infographic.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Evaluation of AI Output

**What is correct:**
- The three-panel structure is clear and non-technical audiences can understand it.
- Decision Tree as "interpretable" and Random Forest as "high accuracy" is accurate.
- The pros and cons are factually correct.

**What AI tends to oversimplify:**
- AI often says "Logistic Regression is less accurate" without noting it can match tree models on linearly separable data.
- AI may understate the fact that Random Forest can still be partially explained via feature importance.

**Corrections applied:**
- Added actual metrics from our loan dataset run.
- Added "When to use" context based on the bank scenario in Part A.
- Added that Random Forest used 200 trees with log2 features (from best params).
- Clarified that credit_score dominates both default (0.73) and permutation importance (0.43).

## Feature Importance Insight (from Part A results)

| Feature | Default Importance | Permutation Importance |
|---|---|---|
| credit_score | 0.7264 | 0.4334 |
| employment_years | 0.0910 | 0.0458 |
| debt_to_income | 0.0739 | 0.0293 |
| annual_income | 0.0599 | 0.0133 |
| loan_amount | 0.0332 | 0.0017 |
| num_credit_cards | 0.0157 | 0.0001 |

Both methods agree: credit_score is by far the most important feature.
Default importance inflates the score slightly because it counts split frequency, while permutation importance is a truer measure of how much removing a feature hurts ROC-AUC.

## Bank Deployment Recommendation (1-Paragraph)

The bank should deploy Random Forest as the primary prediction model, using the Decision Tree as a regulatory explanation layer.
Random Forest achieves 0.9483 accuracy and 0.9907 ROC-AUC, clearly outperforming the Decision Tree (0.8767 / 0.9555) and Logistic Regression baseline.
The three Decision Tree rules provide clear, auditable logic for regulators without requiring full model disclosure.
The most critical feature is credit_score, confirmed by both default and permutation importance methods, which aligns well with standard banking risk frameworks.
Extra Trees achieved marginally better performance (0.9567 / 0.9940) but the tuned Random Forest is more controllable for production deployments.
