# -*- coding: utf-8 -*-
import unittest
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.helpers.parameters import format_value


class TestParameters(ExtTestCase):
    def test_format_value(self):
        self.assertEqual("3", format_value(3))
        self.assertEqual("'3'", format_value("3"))


if __name__ == "__main__":
    unittest.main()
