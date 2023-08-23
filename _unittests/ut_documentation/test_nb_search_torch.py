# -*- coding: utf-8 -*-
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import (
    add_missing_development_version,
    skipif_circleci,
    skipif_appveyor,
    skipif_travis,
)
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
import mlinsights


class TestNotebookSearchTorch(unittest.TestCase):
    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    @skipif_travis("torch is ")
    @skipif_appveyor("torch misses a DLL")
    @skipif_circleci("torch is not installed")
    def test_notebook_search_images(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        self.assertTrue(mlinsights is not None)
        folder = os.path.join(
            os.path.dirname(__file__), "..", "..", "_doc", "notebooks", "explore"
        )
        test_notebook_execution_coverage(
            __file__,
            "torch",
            folder,
            "mlinsights",
            copy_files=["data/dog-cat-pixabay.zip"],
            fLOG=fLOG,
        )


if __name__ == "__main__":
    unittest.main()
