# -*- coding: utf-8 -*-
"""
@brief      test log(time=16s)
"""

import sys
import os
import unittest
import sklearn
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import add_missing_development_version
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.texthelper import compare_module_version


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


class TestNotebookPiecewise(unittest.TestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    @unittest.skipIf(compare_module_version(sklearn.__version__, "0.21") < 0,
                     reason="This notebook uses Criterion API changed in 0.21")
    def test_notebook_piecewise(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertTrue(src.mlinsights is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks", "sklearn_c")
        test_notebook_execution_coverage(
            __file__, "piecewise", folder, 'mlinsights', fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
