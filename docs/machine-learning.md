# Machine Learning – Detailed Answers

## Q1. IPL Playoff Prediction – Logistic Regression

### (a) Handling the missing "Mean Batsmen Played"
- Small dataset (n=5) and near-symmetric batsmen counts ⇒ simple, transparent imputation is safest.
- **Mean/median imputation** on observed batsmen counts (7,5,8,4) → mean = 6.0, median = 6.0. Either keeps scale and avoids dropping a row. Chosen: **mean = 6.0**.
- For production, a quick cross-check with k-NN (using runs+wickets) could be used, but for this exam-sized table, mean is adequate and reproducible.

### (b) Z-score normalization (Total Runs, Total Wickets)
Data after batsmen imputation:

| Team | Batsmen | Total Runs | Total Wkts | Qualified |
| --- | --- | --- | --- | --- |
| A | 7 | 820 | 40 | 1 |
| B | 5 | 840 | 44 | 0 |
| C | 6 | 860 | 48 | 1 |
| D | 8 | 880 | 52 | 1 |
| E | 4 | 900 | 56 | 0 |

Means and std (population):
- Runs: mean 860, σ ≈ 28.284.
- Wkts: mean 48, σ ≈ 5.657.

Z-scores ( (x-μ)/σ ):

| Team | Runs z | Wkts z |
| --- | --- | --- |
| A | -1.414 | -1.414 |
| B | -0.707 | -0.707 |
| C | 0 | 0 |
| D | 0.707 | 0.707 |
| E | 1.414 | 1.414 |

We keep Batsmen unscaled so weights stay interpretable across two normalized features and one raw feature.

### (c) One batch-GD iteration (logistic)
Design matrix X columns: [1, Batsmen, Runs z, Wkts z]. Initial θ = [0.1, 0.1, 0.1, 0.1], α = 0.01.

Sigmoid scores (ŷ):
- A: z=0.5172 → ŷ=0.6265
- B: z=0.4586 → ŷ=0.6126
- C: z=0.7000 → ŷ=0.6682
- D: z=1.0414 → ŷ=0.7391
- E: z=0.7828 → ŷ=0.6863

Errors (ŷ−y): [-0.3735, 0.6126, -0.3318, -0.2609, 0.6863].

Gradient (1/m · Xᵀ(ŷ−y)):
- ∂L/∂θ₀ ≈ 0.0665
- ∂L/∂θ_bats ≈ -0.1769
- ∂L/∂θ_runs ≈ 0.1761
- ∂L/∂θ_wkts ≈ 0.1761

Updated weights: θ ← θ − α·grad
- θ₀ ≈ 0.09933
- θ_bats ≈ 0.10177
- θ_runs ≈ 0.09824
- θ_wkts ≈ 0.09824

### (d) Cross-Entropy loss after first step
Average log loss ≈ 0.657.

### (e) Predicted P(playoffs) for Team D after update
z ≈ 1.0524 → ŷ ≈ 0.741. Interpretation: Team D has ~74% modeled chance to qualify after one gradient update given these features.

## Q2. Bias–Variance & Overfitting
- Training RMSE ↓ while validation RMSE stays high ⇒ **low bias, high variance** (overfitting).
- Likely cause: model too complex relative to signal/noise; weak regularization; or validation not i.i.d. with train.
- Fix: simplify/regularize (L2/early-stop), add dropout-style noise, or add cross-validation with feature selection; also ensure proper shuffle/stratification.

## Q3. Decision Tree – Entropy/Information Gain
Full dataset Purchased counts: Yes=5, No=3.
- Entropy = −(5/8)log₂(5/8) − (3/8)log₂(3/8) ≈ **0.95 bits**.

Young branch (rows with Age=Young: 3 Yes, 1 No):
- Entropy ≈ −(3/4)log₂(3/4) − (1/4)log₂(1/4) ≈ **0.81 bits** → not pure.
- Next split: **Income** makes Low (No), Medium (Yes, Yes), High (Yes) → all pure; better than Occupation where Student remains mixed.

Occupation vs Income (overall): Income levels correlate cleanly in the Young subset; Occupation leaves a mixed Student group. So **Income** separates Yes/No more effectively.

New customer (Young, Low, Professional): Majority in Young subset is Yes (3/4), so classify **Yes** by majority vote.

## Q4. Posterior-Prob evaluation (t = 0.5)
### Confusion matrices
- **M1**: TP=3, FP=1, FN=2, TN=4.
- **M2**: TP=1, FP=1, FN=4, TN=4.

### Metrics
- **M1**: Precision 0.75, Recall 0.60, F1 ≈ 0.67.
- **M2**: Precision 0.50, Recall 0.20, F1 ≈ 0.29.

M1 dominates on recall and F1 for the positive-focused objective.

## Q5. Regression Model Interpretation
Model: Price = 50 + 200·Area + 5·Age.
- Coeff 200: each extra unit of area adds 200 (holding age). Strongest driver.
- Coeff 5: each extra unit of age adds 5 (minor effect).
- Unit change Area m² → ft² (1 m² ≈ 10.76 ft²): slope rescales to ≈ 18.6 per ft²; intercept unchanged numerically but now expressed with ft² units—predictions stay physically consistent.
- Gradient clue: large ∂J/∂θ₁, tiny ∂J/∂θ₀ ⇒ slope poorly fit; bias near optimum. Expect larger updates to θ₁ than θ₀.
- Stronger influence: **Area** (|200| ≫ |5|) dominates contribution to price across realistic ranges.

## Quick practice checklist (per topic)
- Logistic regression: recompute one more GD step, try L2 penalty, plot decision boundary in z-scored space.
- Trees: compute full information gain for Income vs Occupation; draw the resulting tree; test a counterexample profile.
- Metrics: sweep thresholds for M1 to build a small PR table; compute macro vs micro F1 with these 10 points.
- Regression: rescale features and see coefficient changes; check gradient norms over a mini-batch.