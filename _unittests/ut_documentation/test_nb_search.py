# -*- coding: utf-8 -*-
"""
@brief      test log(time=13s)
"""

import sys
import os
import unittest
import warnings
from io import StringIO
from contextlib import redirect_stderr
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import add_missing_development_version
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src


import src.mlinsights


class TestNotebookSearch(unittest.TestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    def test_notebook_search_images(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        with redirect_stderr(StringIO()):
            try:
                from keras.applications.mobilenet import MobileNet
                assert MobileNet is not None
            except SyntaxError as e:
                warnings.warn("tensorflow is probably not available yet on python 3.7: {0}".format(e))
                return

        self.assertTrue(src.mlinsights is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(__file__, "search", folder, 'mlinsights',
                                         copy_files=["data/dog-cat-pixabay.zip"], fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
