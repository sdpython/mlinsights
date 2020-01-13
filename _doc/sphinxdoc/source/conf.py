# -*- coding: utf-8 -*-
"""
Configuration for the documntation.
"""
import sys
import os
import sphinx_readable_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "mlinsights", "Xavier Dupré", 2019,
                     "readable", sphinx_readable_theme.get_html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/mlinsights/issues/%s', 'issue')),
                     title="mlinsights", book=True)

blog_root = "http://www.xavierdupre.fr/app/mlinsights/helpsphinx/"

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css'],
}

html_logo = "phdoc_static/project_ico.png"

html_sidebars = {}

language = "en"

mathdef_link_only = True

custom_preamble = """\n
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\loinormale}[2]{{\\cal N}\\pa{#1,#2}}
\\newcommand{\\independant}[0]{\\;\\makebox[3ex]
{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]
{\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
\\newcommand{\\esp}{\\mathbb{E}}
\\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
\\newcommand{\\loi}[0]{{\\cal L}}
\\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\dans}[0]{\\rightarrow}
\\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
\\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
\\newcommand{\\loimultinomiale}[1]{{\\cal M}\\pa{#1}}
\\newcommand{\\trace}[1]{tr\\pa{#1}}
\\newcommand{\\abs}[1]{\\left|#1\\right|}
"""

# \\usepackage{eepic}
imgmath_latex_preamble += custom_preamble
latex_elements['preamble'] += custom_preamble

epkg_dictionary.update({
    'BLAS': 'http://www.netlib.org/blas/explore-html',
    'bootstrap': 'https://en.wikipedia.org/wiki/Bootstrapping_(statistics)',
    'CountVectorizer': 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html',
    'cython': 'https://cython.org/',
    'decision tree': 'https://en.wikipedia.org/wiki/Decision_tree',
    'DOT': 'https://en.wikipedia.org/wiki/DOT_(graph_description_language)',
    'GIL': 'https://wiki.python.org/moin/GlobalInterpreterLock',
    'PEP-0311': 'https://www.python.org/dev/peps/pep-0311/',
    'Iris': 'http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html',
    'LAPACK': 'http://www.netlib.org/lapack/explore-html',
    'Lapack documentation': 'http://www.netlib.org/lapack/explore-html',
    'L1': 'https://en.wikipedia.org/wiki/Norm_(mathematics)#Absolute-value_norm',
    'L2': 'https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm',
    'keras': 'https://keras.io/',
    'MLPClassifier': 'http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html',
    'MLPRegressor': 'http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html',
    'nogil': 'https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#releasing-the-gil',
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/',
               ('http://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.html', 1),
               ('http://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.{1}.html', 2)),
    'PCA': 'https://en.wikipedia.org/wiki/Principal_component_analysis',
    'py-spy': 'https://github.com/benfred/py-spy',
    'RandomForestRegressor': 'http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
    'REST API': 'https://en.wikipedia.org/wiki/Representational_state_transfer',
    'sklearn': ('http://scikit-learn.org/stable/',
                ('http://scikit-learn.org/stable/modules/generated/{0}.html', 1),
                ('http://scikit-learn.org/stable/modules/generated/{0}.{1}.html', 2)),
    'statsmodels': 'https://www.statsmodels.org/stable/index.html',
    't-SNE': 'https://lvdmaaten.github.io/tsne/',
    'TfidfVectorizer': 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html',
    'torch': 'https://pytorch.org/',
    'tqdm': 'https://github.com/tqdm/tqdm',
    'TSNE': 'https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html',
})

nblinks = {
    'alter_pipeline_for_debugging': 'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/mlinsights/helpers/pipeline.html#mlinsights.helpers.pipeline.alter_pipeline_for_debugging',
}
