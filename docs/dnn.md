# Deep Neural Networks & ML Fundamentals – Detailed Answers

> The paper leaves some parameter values blank (certain weights/logits). Where numbers are missing, I give the full formulas so you can plug in the provided values directly.

## Q1. Perceptron & Spam Classification

### (a) Single update
- Weighted sum: \(z = w_0\cdot 1 + w_1 x_1 + w_2 x_2\). With the stated example \(x_1=1, x_2=2\), \(z = w_0 + w_1 + 2w_2\).
- Step output: \(\hat{y} = \mathbf{1}[z \ge 0]\).
- If misclassified (\(\hat{y} \ne y\)), update: \(w_j \leftarrow w_j + \eta (y-\hat{y}) x_j\) for \(j=0,1,2\) using \(x_0=1\).

### (b) Spam feature weight reading
- Largest |weight| after the bias indicates strongest influence. With \(w=[0.2, 0.8, 0.9, -0.5]^T\), the “links” feature (0.9) is most indicative of spam; the negative weight on length implies longer emails decrease spam likelihood.
- Plateau at 75% accuracy suggests data not linearly separable; a single perceptron cannot surpass the best linear boundary. Need more expressive model (e.g., kernel, MLP) or better features.

### (c) Generalization comparison
- Approach A (5 curated features) is lower dimensional → lower variance; with 500 samples, it likely generalizes better and is interpretable.
- Approach B (50 frequent-word counts) is high dimensional relative to 500 samples → higher overfitting risk unless regularized. Same train accuracy suggests possible variance problem.

### (d) Code blanks (perceptron)
- Blank 1: `np.dot(weights, X[i])`
- Blank 2: `y[i] - y_pred`
- Blank 3: `learning_rate * error * X[i]`
- Blank 4: `np.where(np.dot(X, w) >= 0, 1, 0)`
- Blank 5: `(preds == y).mean()`

## Q2. Linear Regression & Gradient Descent

### (a) One batch-GD step (given table)
Data: (x, y) = (10,150), (20,250) with bias 1; weights \(w_0=50, w_1=8\), lr=0.01.
- Predictions: \(\hat{y} = 50 + 8x\) ⇒ 130, 210.
- Errors: −20, −40. MSE = \(\tfrac{1}{2}[(−20)^2 + (−40)^2] = 1000\) (using factor 1/2m with m=2 → loss 500 if you use 1/2m convention). Here shown as average of squared errors without 1/2m: 1000.
- Gradient (using 1/m):
  - \(\partial/\partial w_0 = (1/2)(−20 + −40) = −30\).
  - \(\partial/\partial w_1 = (1/2)(−20·10 + −40·20) = −500\).
- Update: \(w_0 \leftarrow 50 - 0.01(−30) = 50.30\); \(w_1 \leftarrow 8 - 0.01(−500) = 13.00\).

### (b) Code blanks (linear GD)
- Blank 1: `X @ weights`
- Blank 2: `((y_pred - y) ** 2).mean()`
- Blank 3: `(X.T @ error) / N`
- Blank 4: `weights - lr * gradient`
- Blank 5: `train_linear(X, y)`

### (c) Imbalanced diagnostics
- Accuracy: \((40+855)/1000 = 0.895\) ≈ 89.5%. A naive “always healthy” gives 95%—so accuracy is misleading.
- For life-threatening diseases, recall is critical: lowering threshold to 0.3 increases TP and FP, reduces FN; more patients flagged, fewer misses.

### (d) Fraud model choice
- Model B catches 75 more frauds (75 × 100 loss avoided = 7,500) vs Model A. Extra investigations cost: more positives, but given strong fraud gain, Model B is preferable if operational cost of investigations doesn’t exceed saved fraud losses. Added complexity is justified by higher recall on rare costly class.

## Q3. Logistic Regression

### (a) Single example update (loan)
Given \(w_0=0, w_1=0.6, w_2=0.8, x=(1, 0.7, 0.5), y=1\):
- \(z = 0 + 0.6·0.7 + 0.8·0.5 = 0.82\).
- \(\hat{y} = \sigma(z) = 1/(1+e^{-0.82}) \approx 0.694\).
- Error: \(\hat{y} - y \approx -0.306\).
- Gradient (for one sample): \((\hat{y}-y)x = [-0.306, -0.2142, -0.153]\).
- Update with lr=0.1: \(w \leftarrow w - 0.1 \cdot \text{grad}\) ⇒ \(w_0 \approx 0.0306, w_1 \approx 0.6214, w_2 \approx 0.8153\).

### (b) Code blanks (mini-batch logistic)
- Blank 1: `1 / (1 + np.exp(-z))`
- Blank 2: `sigmoid(z)`
- Blank 3: `(X_batch.T @ (y_pred - y_batch)) / len(y_batch)`
- Blank 4: `weights - lr * gradient`
- Blank 5: `(sigmoid(np.dot(X, weights)) >= 0.5).astype(int)`

## Q4. Softmax & Confusion Matrix

### (a) Softmax example
For logits \(\ell = (\ell_0, \ell_1, \ell_2)\), softmax \(p_i = e^{\ell_i}/\sum_j e^{\ell_j}\). CCE loss for true class 1: \(-\log p_1\). Predicted class = argmax of logits (or probabilities). Correctness: check if argmax is 1.

### (b) Code blanks (softmax training)
- Blank 1: `exp_Z / np.sum(exp_Z, axis=1, keepdims=True)`
- Blank 2: `np.dot(X_batch, W)`
- Blank 3: `(X_batch.T @ (Y_pred - Y_batch)) / len(Y_batch)`
- Blank 4: `W - lr * gradient`
- Blank 5: `np.argmax(Z, axis=1)`

### (c) Class-wise metrics
- For Bird: Precision = TP/(TP+FP); Recall = TP/(TP+FN). Use the provided confusion counts once filled. If 40 of 50 bird errors are predicted Dog, model confuses bird ↔ dog features; overall accuracy (e.g., 88%) may hide poor minority-class performance.

### (d) Imbalanced medical case
- Prefer Model B despite lower overall accuracy because TB recall is far higher (78% vs 45%); missing TB has high clinical risk. Ethical deployment prioritizes recall on rare but critical class over aggregate accuracy.

## Q5. 2-layer DFNN

### (a) Forward pass (symbolic)
Given X in \(\mathbb{R}^{1\times2}\), parameters (W1\(\in\mathbb{R}^{2\times2}\), b1\(\in\mathbb{R}^{1\times2}\), W2\(\in\mathbb{R}^{2\times1}\), b2\(\in\mathbb{R}\)):
1. \(Z_1 = X W_1 + b_1\).
2. \(A_1 = \text{ReLU}(Z_1)\).
3. \(Z_2 = A_1 W_2 + b_2\).
4. \(\hat{y} = \sigma(Z_2)\); BCE loss for label 1: \(-\log \hat{y}\).

### (b) Code blanks (DFNN)
- Blank 1: `np.maximum(0, Z)`
- Blank 2: `relu(Z1)`
- Blank 3: `A1.T @ dZ2 / N`
- Blank 4: `dZ2 @ W2.T`
- Blank 5: `X.T @ dZ1 / N`

## Quick practice (DL/ML)
- Perceptron: try a non-separable XOR minibatch to observe non-convergence.
- Logistic: sweep thresholds and plot PR for the fraud example; compute cost-sensitive metrics.
- Softmax: plug logits (e.g., [2,0,1]) and compute probabilities and loss for class 1.
- DFNN: implement a tiny 2-2-1 network on AND/OR; verify gradients with finite differences.