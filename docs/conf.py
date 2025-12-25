import os
import sys

project = "WILP Mid Sem Answers"
author = "Auto-generated with MyST"

extensions = [
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "furo"
html_static_path = ["_static"]
html_title = project

sys.path.insert(0, os.path.abspath(".."))