import unittest
from mlinsights.ext_test_case import ExtTestCase
from mlinsights._config import CYTHON_VERSION


class TestCheckVersion(ExtTestCase):
    def test_version_cython(self):
        self.assertVersionGreaterOrEqual(CYTHON_VERSION, "3.0.5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
