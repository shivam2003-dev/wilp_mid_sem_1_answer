# WILP Mid-Sem Answers

MyST + Sphinx site with detailed solutions, intuition, practice links, and auto-deployed GitHub Pages.

## Local preview

```bash
pip install -r requirements.txt
sphinx-build -b html docs _build/html
open _build/html/index.html
```

## Structure
- docs/index.md – landing and navigation
- docs/machine-learning.md – answers for Machine Learning mid-sem
- docs/ml-practice.md – practice sets, videos, and references
- .github/workflows/gh-pages.yml – Pages CI/CD