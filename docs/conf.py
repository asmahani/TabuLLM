"""Sphinx configuration for TabuLLM's documentation site.

Built with autodoc + napoleon (docstrings are NumPy-style) and autosummary
(one stub page per object exported in ``tabullm.__all__`` -- see api.md).
Narrative pages (index, examples, paper) are authored in MyST Markdown via
myst_parser. Modeled on the MetaCausal documentation setup.
"""

from __future__ import annotations

import tabullm

project = "TabuLLM"
copyright = "2024, Alireza S. Mahani, Mansour T.A. Sharabiani"
author = "Alireza S. Mahani, Mansour T.A. Sharabiani"
version = tabullm.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# napoleon (all TabuLLM docstrings use NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Strip the >>> / ... REPL prompts from copied snippets so pasted examples
# are directly runnable -- matches numpy/scipy/pandas/scikit-learn convention.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# dollarmath: the README uses inline $...$ math for the GMM log-joint formula.
myst_enable_extensions = ["colon_fence", "dollarmath"]
# Auto-generate slugged anchors for headings up to h3, so in-page links like
# `[Quick Example](#quick-example)` from the mirrored README resolve.
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "TabuLLM"
html_static_path = ["_static"]

# ClusterExplainer's default-prompt class attributes (`parser`,
# `CONTEXT_SAFETY_MARGIN`, `prompt_*`) are internal implementation details,
# not public API -- customization happens via the `custom_prompts`
# constructor argument, documented in the class docstring. Autodoc would
# otherwise render their full repr (a giant PromptTemplate dump, including
# the JSON-schema format instructions) because a class attribute counts as
# "documented" whenever its *value's type* has a docstring -- e.g. a float
# or a PromptTemplate instance -- even with no `#:` comment of its own, so
# `undoc-members: False` alone doesn't filter them out.
_CLUSTER_EXPLAINER_INTERNAL_ATTRS = frozenset({
    "parser",
    "CONTEXT_SAFETY_MARGIN",
    "prompt_label_direct",
    "prompt_label_from_summaries",
    "prompt_summarize_observations",
    "prompt_combine_summaries",
    "prompt_synthesize",
})


def _skip_internal_class_attrs(app, what, name, obj, skip, options):
    if name in _CLUSTER_EXPLAINER_INTERNAL_ATTRS:
        return True
    # Returning `None` (not `skip`) here matters: autosummary's own
    # autodoc-skip-member call always passes `skip=False`, so returning
    # it back verbatim would be interpreted as "force-include this member"
    # for every member, bypassing the default private/dunder filtering.
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_internal_class_attrs)
