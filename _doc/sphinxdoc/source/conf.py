#-*- coding: utf-8 -*-
import sys
import os
import datetime
import re
import guzzle_sphinx_theme


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.split(__file__)[0],
            "..",
            "..",
            "..",
            "..",
            "pyquickhelper",
            "src")))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet
set_sphinx_variables(__file__, "mlinsights", "Xavier Dupré", 2017,
                     "guzzle_sphinx_theme", guzzle_sphinx_theme.html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/mlinsights/issues/%s', 'issue')),
                     title="mlinsights", book=True)

blog_root = "http://www.xavierdupre.fr/app/mlinsights/helpsphinx/"
extensions.append("guzzle_sphinx_theme")
html_theme_options['project_nav_name'] = 'mlinsights'
html_theme_options['touch_icon'] = 'project_ico.ico'


html_logo = "project_ico.png"

html_sidebars = {}

language = "en"

mathdef_link_only = True

epkg_dictionary['REST API'] = "https://en.wikipedia.org/wiki/Representational_state_transfer"
