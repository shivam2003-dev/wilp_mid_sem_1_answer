# WILP Mid-Sem Answers

Jupyter Book (MyST) site with detailed solutions, intuition, practice links, and auto-deployed GitHub Pages.

## Local preview

```bash
pip install -r requirements.txt
jupyter-book build docs --path-output _build
open _build/html/index.html
```

## Structure
- docs/index.md – landing and navigation
- docs/machine-learning.md – answers for Machine Learning mid-sem
- docs/ml-practice.md – practice sets, videos, and references
- docs/math-ml.md – math for ML answers
- docs/dnn.md – deep neural nets fundamentals answers
- docs/_config.yml and docs/_toc.yml – Jupyter Book config and ToC
- .github/workflows/gh-pages.yml – Pages CI/CD