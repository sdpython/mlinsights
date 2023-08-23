# -*- coding: utf-8 -*-
import os
import unittest
import warnings
from io import StringIO
from contextlib import redirect_stderr
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import add_missing_development_version
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
import mlinsights


class TestNotebookSearchKeras(unittest.TestCase):
    def setUp(self):
        add_missing_development_version(["jyquickhelper"], __file__, hide=True)

    def test_notebook_search_images(self):
        fLOG(__file__, self._testMethodName, OutputPrint=__name__ == "__main__")

        with redirect_stderr(StringIO()):
            try:
                from keras.applications.mobilenet import (
                    MobileNet,
                )  # pylint: disable=E0401,E0611

                assert MobileNet is not None
            except (SyntaxError, ModuleNotFoundError, AttributeError, ImportError) as e:
                warnings.warn(
                    f"tensorflow is probably not available yet on python 3.7: {e}"
                )
                return

        self.assertTrue(mlinsights is not None)
        folder = os.path.join(
            os.path.dirname(__file__), "..", "..", "_doc", "notebooks", "explore"
        )
        test_notebook_execution_coverage(
            __file__,
            "keras",
            folder,
            "mlinsights",
            copy_files=["data/dog-cat-pixabay.zip"],
            fLOG=fLOG,
        )


if __name__ == "__main__":
    unittest.main()
