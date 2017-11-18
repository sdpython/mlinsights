"""
@brief      test log(time=0s)
"""

import sys
import os
import unittest


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

try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    if "PYQUICKHELPER" in os.environ and len(os.environ["PYQUICKHELPER"]) > 0:
        sys.path.append(os.environ["PYQUICKHELPER"])
    import pyquickhelper as skip_


from pyquickhelper.loghelper import fLOG
from pyquickhelper.filehelper import explore_folder_iterfile
from pyquickhelper.pycode import ExtTestCase


class TestConvertNotebooks(ExtTestCase):

    def test_convert_notebooks(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        try:
            import jyquickhelper as skip___
            rem = None
        except ImportError:
            p = os.path.dirname(src.__file__)
            fLOG("add path", p)
            rem = len(sys.path) - 1
            sys.path.append(p)
        from pyquickhelper.ipythonhelper import upgrade_notebook, remove_execution_number
        if rem:
            del sys.path[rem]

        fold = os.path.abspath(os.path.dirname(__file__))
        fold2 = os.path.normpath(
            os.path.join(fold, "..", "..", "_doc", "notebooks"))
        for nbf in explore_folder_iterfile(fold2, pattern=".*[.]ipynb"):
            t = upgrade_notebook(nbf)
            if t:
                fLOG("modified", nbf)
            # remove numbers
            remove_execution_number(nbf, nbf)

        fold2 = os.path.normpath(os.path.join(fold, "..", "..", "_unittests"))
        for nbf in explore_folder_iterfile(fold2, pattern=".*[.]ipynb"):
            t = upgrade_notebook(nbf)
            if t:
                fLOG("modified", nbf)


if __name__ == "__main__":
    unittest.main()
