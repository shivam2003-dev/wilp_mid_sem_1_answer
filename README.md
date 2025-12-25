# WILP Mid-Sem Answers (Jupyter Book)

This repo is set up as a Jupyter Book and deploys to GitHub Pages via GitHub Actions.

## Local build

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter-book build book
```

The built HTML will be in `book/_build/html/`.

## GitHub Pages

- Deployment is handled by `.github/workflows/deploy.yml`.
- It publishes the built book to the `gh-pages` branch.
