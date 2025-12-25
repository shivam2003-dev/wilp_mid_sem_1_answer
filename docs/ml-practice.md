# Practice & References – Machine Learning

## Targeted practice problems
- Logistic regression: build a tiny notebook to run 3 GD steps on the IPL table; add L2=0.1 and compare weights.
- Decision trees: compute information gain for Income vs Occupation on the 8-row dataset; then prune by minimum samples per leaf.
- Metrics: using the 10-instance table, vary threshold {0.3, 0.5, 0.7}; plot precision/recall/F1; identify the operating point that maximizes F1.
- Regression scaling: convert area m²→ft² and refit; show coefficient scaling but unchanged predictions.

## External practice sets
- Binary classification toy sets (logistic, trees): UCI “Bank Marketing”, Kaggle “Titanic”.
- Imbalanced metrics practice: “Credit Card Fraud” (Kaggle) for precision/recall/PR curves.
- Small regression: scikit-learn “Boston housing” (educational only), or “California housing” for modern use.

## Video explainers (YouTube)
- StatQuest – Logistic Regression: intuition and math.
- StatQuest – Decision Trees and Entropy.
- Josh Starmer – Precision, Recall, F1 (imbalanced focus).
- 3Blue1Brown – Gradient Descent (visual intuition).

## Suggested reading
- Bishop, *Pattern Recognition and Machine Learning*: Chapters on linear models and classification.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*: bias–variance, model assessment.
- scikit-learn docs: `LogisticRegression`, `DecisionTreeClassifier`, `precision_recall_curve` examples.