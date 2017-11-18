"""
@brief      test log(time=0s)
"""

import sys
import os
import unittest
import re


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

from src.mlinsights import __version__


class TestVersion (unittest.TestCase):

    def test_version(self):
        setup = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "..",
            "setup.py")
        with open(setup, "r") as f:
            c = f.read()
        reg = re.compile("sversion *= \\\"(.*)\\\"")

        f = reg.findall(c)
        if len(f) != 1:
            raise Exception("not only one version")
        self.assertEqual(f[0], __version__)


if __name__ == "__main__":
    unittest.main()
