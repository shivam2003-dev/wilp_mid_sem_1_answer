# WILP Mid-Sem 1 (Dec 2025) — Solved Answers

## Q1. IPL Playoff Prediction (Logistic Regression)

**Training dataset**

| Team | Mean Batsmen Played | Total Runs | Total Wickets | Qualified (y) |
| :--- | ---: | ---: | ---: | ---: |
| A | 7 | 820 | 40 | 1 |
| B | 5 | 840 | 44 | 0 |
| C | ? | 860 | 48 | 1 |
| D | 8 | 880 | 52 | 1 |
| E | 4 | 900 | 56 | 0 |

### (a) Missing value handling
- Use **mean/median imputation** (simple + stable for tiny dataset).
- More complex (KNN/regression) imputation is possible but risks overfitting with only 5 rows.

Mean/median of {7, 5, 8, 4} = 6 → impute Team C batsmen = **6**.

### (b) Z-score normalization (only Runs & Wickets)
Z-score: $z = (x-\mu)/\sigma$.

Runs: values {820,840,860,880,900}
- $\mu_{\text{runs}} = 860$
- $\sigma_{\text{runs}} = \sqrt{\frac{\sum (x-\mu)^2}{5}} = \sqrt{800} \approx 28.2843$

Wickets: values {40,44,48,52,56}
- $\mu_{\text{wkts}} = 48$
- $\sigma_{\text{wkts}} = \sqrt{32} \approx 5.6569$

Updated dataset:

| Team | Batsmen | Runs (z) | Wickets (z) | y |
|---|---:|---:|---:|---:|
| A | 7 | -1.4142 | -1.4142 | 1 |
| B | 5 | -0.7071 | -0.7071 | 0 |
| C | 6 | 0.0000 | 0.0000 | 1 |
| D | 8 | 0.7071 | 0.7071 | 1 |
| E | 4 | 1.4142 | 1.4142 | 0 |

### (c) One batch gradient descent iteration
Assume $x = [1, x_1, x_2, x_3]$ where:
- $x_1$ = Mean batsmen
- $x_2$ = Runs(z)
- $x_3$ = Wickets(z)

Sigmoid: $\sigma(z)=\frac{1}{1+e^{-z}}$.

Given initial $\theta=[0.1,0.1,0.1,0.1]$, $\alpha=0.01$, $m=5$.

Compute $h=\sigma(\theta^T x)$ and errors $(h-y)$:
- A: $z=0.5172$, $h\approx 0.6265$, $h-y\approx -0.3735$
- B: $z=0.4586$, $h\approx 0.6127$, $h-y\approx 0.6127$
- C: $z=0.7000$, $h\approx 0.6682$, $h-y\approx -0.3318$
- D: $z=1.0414$, $h\approx 0.7391$, $h-y\approx -0.2609$
- E: $z=0.7828$, $h\approx 0.6863$, $h-y\approx 0.6863$

Gradient (cross-entropy):
$$
\nabla J(\theta)=\frac{1}{m}\sum_{i=1}^m (h^{(i)}-y^{(i)})x^{(i)}
$$

Gradient components (approx):
- $g_0\approx 0.06656$
- $g_1\approx -0.17676$
- $g_2\approx 0.17620$
- $g_3\approx 0.17620$

Update: $\theta := \theta - \alpha g$
- $\theta_0 \approx 0.09933$
- $\theta_1 \approx 0.10177$
- $\theta_2 \approx 0.09824$
- $\theta_3 \approx 0.09824$

### (d) Log loss after first iteration
Using updated $\theta$, predicted probabilities:
- A: $\approx 0.6304$
- B: $\approx 0.6152$
- C: $\approx 0.6704$
- D: $\approx 0.7412$
- E: $\approx 0.6866$

$$
J = -\frac{1}{m}\sum\big[y\ln(h) + (1-y)\ln(1-h)\big] \approx 0.655
$$

### (e) Team D probability
$P(\text{qualify} \mid D) \approx 0.741$ → about **74.1%** chance.

---

## Q2. Bias–Variance & Overfitting

Observation: training RMSE drops but validation RMSE stays high.
- (a) **Low bias, high variance** (overfitting).
- (b) Likely reason: model too complex / noisy features / still insufficient data for that complexity.
- (c) Fix: **regularization** (L2/L1), simplify model, early stopping, or better features.

---

## Q3. Decision Tree — Entropy & Information Gain

Dataset:

| Agegroup | Income Level | Occupation | Purchased |
| --- | --- | --- | --- |
| Young | Low | Student | No |
| Middle-aged | High | Professional | Yes |
| Young | Medium | Student | Yes |
| Old | Low | Retired | No |
| Young | High | Professional | Yes |
| Middle-aged | Low | Professional | No |
| Old | Medium | Retired | Yes |
| Young | Medium | Professional | Yes |

### (i) Entropy of Purchased
Yes = 5, No = 3 (total 8)

$$
H(S)=-\frac{5}{8}\log_2\frac{5}{8}-\frac{3}{8}\log_2\frac{3}{8} \approx 0.9544
$$

### (ii) Entropy for Young branch
Young rows = 4 → Yes=3, No=1

$$
H(\text{Young})=-\frac{3}{4}\log_2\frac{3}{4}-\frac{1}{4}\log_2\frac{1}{4} \approx 0.8113
$$

Not pure. Next split: **Income Level** makes the Young subset pure:
- Low → {No}
- Medium → {Yes, Yes}
- High → {Yes}

### (iii) Occupation vs Income Level
Income Level separates better here:
- Low income → all **No**
- Medium/High → all **Yes**

### (iv) Majority vote in Young subset
Young subset majority is **Yes (3/4)** → classify as **Purchased = Yes**.

---

## Q4. Model Evaluation (Posterior Probabilities)

Threshold $t=0.5$.

| Instance | True | M1 P(+) | M2 P(+) |
| :--- | :---: | ---: | ---: |
| 1 | + | 0.73 | 0.61 |
| 2 | + | 0.69 | 0.03 |
| 3 | - | 0.44 | 0.68 |
| 4 | - | 0.55 | 0.31 |
| 5 | + | 0.67 | 0.45 |
| 6 | + | 0.47 | 0.09 |
| 7 | - | 0.08 | 0.38 |
| 8 | - | 0.15 | 0.05 |
| 9 | + | 0.45 | 0.01 |
| 10 | - | 0.35 | 0.04 |

### (a) Confusion matrices
**M1** predicts + for {1,2,4,5}
- TP=3 (1,2,5), FP=1 (4), TN=4 (3,7,8,10), FN=2 (6,9)

| M1 | Pred + | Pred - |
|---|---:|---:|
| Actual + | 3 | 2 |
| Actual - | 1 | 4 |

**M2** predicts + for {1,3}
- TP=1 (1), FP=1 (3), TN=4 (4,7,8,10), FN=4 (2,5,6,9)

| M2 | Pred + | Pred - |
|---|---:|---:|
| Actual + | 1 | 4 |
| Actual - | 1 | 4 |

### (b) Precision, Recall, F1
M1:
- Precision = $3/(3+1)=0.75$
- Recall = $3/(3+2)=0.60$
- F1 $=\frac{2PR}{P+R}\approx 0.6667$

M2:
- Precision = $1/(1+1)=0.50$
- Recall = $1/(1+4)=0.20$
- F1 $\approx 0.2857$

---

## Q5. Regression Model Interpretation

Model:
$$
\text{Price} = 50 + 200\,(\text{Area}) + 5\,(\text{Age})
$$

- (a) Coefficients:
  - +200: holding Age fixed, +1 unit Area increases price by 200.
  - +5: holding Area fixed, +1 year Age increases price by 5.

- (b) Changing units m² → ft² rescales coefficient.
  - $1\,\text{m}^2 \approx 10.7639\,\text{ft}^2$ so new coefficient $\approx 200/10.7639 \approx 18.58$ (per ft²).

- (B) If $\partial J/\partial \theta_1$ is large while $\partial J/\partial \theta_0$ is near 0:
  - $\theta_1$ needs a significant update; $\theta_0$ is near optimum.

- (C) Stronger influence: **Area** (much larger coefficient magnitude, assuming comparable scaling).
