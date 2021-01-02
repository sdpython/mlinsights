# -*- coding: utf-8 -*-
"""
@brief      test log(time=43s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import add_missing_development_version
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
from pyquickhelper.texthelper import compare_module_version
import mlinsights


class TestNotebookTargetPredictor(unittest.TestCase):

    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    def test_notebook_target_predictor(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertTrue(mlinsights is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks", "sklearn")
        try:
            test_notebook_execution_coverage(
                __file__, "sklearn_transformed_target",
                folder, 'mlinsights', fLOG=fLOG)
        except AttributeError as e:
            if compare_module_version(sklver, "0.24") < 0:
                return
            raise e


if __name__ == "__main__":
    unittest.main()
