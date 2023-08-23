# -*- coding: utf-8 -*-
import os
import unittest
import sklearn
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import add_missing_development_version, skipif_appveyor
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.texthelper import compare_module_version
import mlinsights


class TestNotebookPiecewise(unittest.TestCase):
    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    @unittest.skipIf(
        compare_module_version(sklearn.__version__, "0.21") < 0,
        reason="This notebook uses Criterion API changed in 0.21",
    )
    @skipif_appveyor("too long")
    def test_notebook_piecewise(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        self.assertTrue(mlinsights is not None)
        folder = os.path.join(
            os.path.dirname(__file__), "..", "..", "_doc", "notebooks", "sklearn_c"
        )
        test_notebook_execution_coverage(
            __file__, "piecewise", folder, "mlinsights", fLOG=fLOG
        )


if __name__ == "__main__":
    unittest.main()
