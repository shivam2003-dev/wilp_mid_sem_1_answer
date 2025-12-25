# DNN / ML Fundamentals (Dec 2025) — Solved Answers

Some numeric values (perceptron initial weights, softmax logits, confusion matrix entries, DFNN parameters) are not present in the extracted text available in this workspace. Where those are missing, this page provides the correct method + final formulas; if you share the missing numbers, I can fill the exact numeric results.

---

## Q1. Perceptron & Spam Classification

### Q1(a) Weighted sum, prediction, and update
For a 2-feature perceptron with bias:
- Input: $x=[x_0,x_1,x_2]=[1,1,2]$
- Weights: $w=[w_0,w_1,w_2]$

1) Weighted sum:
$$z=w_0\cdot 1+w_1\cdot 1+w_2\cdot 2$$

2) Step prediction:
$$\hat y = \begin{cases}1 & z\ge 0\\0 & z<0\end{cases}$$

3) If misclassified, perceptron update (learning rate $\eta=0.1$):
$$w \leftarrow w + \eta (y-\hat y) x$$
So:
- $w_0 \leftarrow w_0 + \eta (y-\hat y)\cdot 1$
- $w_1 \leftarrow w_1 + \eta (y-\hat y)\cdot 1$
- $w_2 \leftarrow w_2 + \eta (y-\hat y)\cdot 2$

### Q1(b) Spam feature + limitation
Given $w=[0.2,0.8,0.9,-0.5]^T$ for (bias, suspicious_words, links, length):
- Strongest spam indicator is **links** (largest positive magnitude: 0.9).
- Stalling at 75% suggests **not linearly separable** data (or label noise/feature insufficiency). A single-layer perceptron can only learn a linear boundary.

### Q1(c) Generalization (5 features vs 50 features)
With 500 reviews and equal training accuracy:
- **Approach A** more likely generalizes better: fewer features → lower variance/overfitting risk + more interpretable.
- **Approach B** higher dimensional bag-of-words can overfit with limited data unless regularized.

### Q1(d) Perceptron code blanks
```python
# Blank 1
np.dot(X[i], weights)

# Blank 2
y[i] - y_pred

# Blank 3
learning_rate * error * X[i]

# Blank 4
np.array([1 if np.dot(x, w) >= 0 else 0 for x in X])

# Blank 5
np.mean(preds == y)
```

---

## Q2. Linear Regression & Gradient Descent

### Q2(a) One batch GD iteration
Data: $(x, y)$ = (10,150), (20,250) with bias term.
Weights: $w_0=50, w_1=8$, learning rate $\eta=0.01$.

Predictions:
- $\hat y_1 = 50 + 8\cdot 10 = 130$
- $\hat y_2 = 50 + 8\cdot 20 = 210$

Errors $e=\hat y-y$:
- $e_1=-20$, $e_2=-40$

MSE:
$$\text{MSE} = \frac{1}{2}\big(e_1^2+e_2^2\big)=\frac{1}{2}(400+1600)=1000$$

Gradient for MSE $=\frac{1}{N}\sum e^2$:
$$\nabla = \frac{2}{N}X^T e$$
Here $N=2$, so factor is 1.
- $\frac{\partial}{\partial w_0}: (-20)+(-40)=-60$
- $\frac{\partial}{\partial w_1}: (-20)\cdot 10+(-40)\cdot 20=-1000$

Update:
- $w_0 \leftarrow 50 - 0.01(-60)=50.6$
- $w_1 \leftarrow 8 - 0.01(-1000)=18.0$

### Q2(b) Linear regression code blanks
```python
# Blank 1
y_pred = X @ weights

# Blank 2
loss = np.mean((y_pred - y) ** 2)

# Blank 3
gradient = (2 / N) * (X.T @ error)

# Blank 4
weights = weights - lr * gradient

# Blank 5
w = train_linear(X, y)
```

### Q2(c) Imbalanced medical classification
Given TP=40, FN=10, FP=95, TN=855, total=1000:
- Accuracy $=(TP+TN)/N=(40+855)/1000=0.895=89.5\%$.
- Misleading because 95% are healthy; always-healthy baseline = $950/1000=95\%$.
- Lowering threshold (0.5→0.3): predicts disease more often → **FP increases**, **FN decreases**.

### Q2(d) Fraud model choice (cost view)
- Model A: catches 300/500 → misses 200 → missed-fraud cost = $200\cdot 100=20000$.
- Model B: catches 375/500 → misses 125 → missed-fraud cost = $125\cdot 100=12500$.
- Model B saves $7500$ in missed-fraud losses.

Investigation cost depends on how many transactions get flagged (not provided). Model B is better as long as it doesn’t add more than $7500/10=750$ extra investigations compared to A.

---

## Q3. Logistic Regression

### Q3(a) One example update
Given $w_0=0$, $w_1=0.6$, $w_2=0.8$, $\eta=0.1$, $x=[1,0.7,0.5]$, $y=1$.

1) $z = 0 + 0.6\cdot 0.7 + 0.8\cdot 0.5 = 0.82$.

2) $\hat y = \sigma(z)=\frac{1}{1+e^{-0.82}}\approx 0.694$.

3) For logistic cross-entropy, per-example gradient: $(\hat y-y)x$.
Here $(\hat y-y)\approx -0.306$.

4) Update: $w \leftarrow w - \eta(\hat y-y)x = w + 0.1\cdot 0.306\cdot x$.
- $w_0\approx 0.0306$
- $w_1\approx 0.6 + 0.0306\cdot 0.7 = 0.62142$
- $w_2\approx 0.8 + 0.0306\cdot 0.5 = 0.81530$

### Q3(b) Logistic code blanks
```python
# Blank 1
1 / (1 + np.exp(-z))

# Blank 2
sigmoid(z)

# Blank 3
(X_batch.T @ (y_pred - y_batch)) / len(y_batch)

# Blank 4
weights - lr * gradient

# Blank 5
(sigmoid(X @ weights) >= 0.5).astype(int)
```

---

## Q4. Softmax & Confusion Matrix

### Q4(a) Softmax + CCE (method)
Given logits $\ell=[\ell_0,\ell_1,\ell_2]$:
$$p_k=\frac{e^{\ell_k}}{\sum_j e^{\ell_j}}$$
CCE for true class 1 (neutral):
$$\text{CCE}=-\log(p_1)$$
Prediction is $\arg\max_k p_k$.

### Q4(b) Softmax code blanks
```python
# Blank 1
exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Blank 2
np.dot(X_batch, W)

# Blank 3
(X_batch.T @ (Y_pred - Y_batch)) / len(idx)

# Blank 4
W - lr * gradient

# Blank 5
np.argmax(Z, axis=1)
```

### Q4(c) Precision/Recall for Bird (method)
From the confusion matrix:
- $TP$ = Bird predicted as Bird
- $FP$ = non-Bird predicted as Bird
- $FN$ = Bird predicted as non-Bird

$$\text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN}$$

If most bird errors go to Dog (40/50), model likely learns bird features that overlap dog-like features; overall 88% accuracy can still hide poor minority-class performance.

### Q4(d) Medical imbalance (TB)
Prefer higher TB recall when missing TB is costly:
- Model B (78% TB recall) is typically preferable clinically despite lower overall accuracy.

---

## Q5. Deep Feedforward Neural Network (DFNN)

### Q5(a) Forward propagation (method)
For hidden layer:
- $Z_1 = XW_1 + b_1$
- $A_1 = \text{ReLU}(Z_1)$
For output:
- $Z_2 = A_1 W_2 + b_2$
- $A_2 = \sigma(Z_2)$
BCE (for $y\in\{0,1\}$):
$$L = -\big[y\log(A_2) + (1-y)\log(1-A_2)\big]$$

### Q5(b) DFNN code blanks
```python
# Blank 1
np.maximum(0, Z)

# Blank 2
relu(Z1)

# Blank 3
(A1.T @ dZ2) / N

# Blank 4
dZ2 @ W2.T

# Blank 5
(X.T @ dZ1) / N
```

### Q5(c) 5-class sentiment DFNN (example design)
One reasonable design:
- Input 1000
- Dense 512 (ReLU)
- Dense 128 (ReLU)
- Dense 5 (Softmax)

Loss: categorical cross-entropy. Training: mini-batch SGD/Adam.

Parameter count:
- 1000→512: $1000\cdot 512 + 512$
- 512→128: $512\cdot 128 + 128$
- 128→5: $128\cdot 5 + 5$

Total = $512000+512 + 65536+128 + 640+5 = 578{,}821$ params.

### Q5(d) Mobile deployment
On phones (4–6GB RAM), prefer **Arch A** (200MB, 5ms). For cloud servers, Arch B may be acceptable if latency/RAM are not constraints.
