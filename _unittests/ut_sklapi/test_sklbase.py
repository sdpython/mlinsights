"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
from pyquickhelper.pycode import ExtTestCase


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

from src.mlinsights.sklapi.sklearn_base import SkBase
from src.mlinsights.sklapi.sklearn_base_learner import SkBaseLearner


class TestSklearnBase(ExtTestCase):

    def test_sklearn_base_parameters(self):
        sk = SkBase(pa1="r", pa2=2)
        p = sk.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk)
        self.assertEqual(r, "SkBase(pa1='r', pa2=2)")
        self.assertTrue(sk.test_equality(sk))

    def test_sklearn_equality(self):
        sk1 = SkBaseLearner(pa1="r", pa2=2)
        p = sk1.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk1)
        self.assertEqual(r, "SkBaseLearner(pa1='r', pa2=2)")
        sk2 = SkBase(pa1="r", pa2=2)
        self.assertFalse(sk1.test_equality(sk2))

    def test_sklearn_compare(self):
        p1 = dict(pa1="r", pa2=2)
        p2 = dict(pa1="r", pa2=2, pa3=4)
        self.assertRaise(lambda: SkBase.compare_params(p1, p2), KeyError)
        self.assertRaise(lambda: SkBase.compare_params(p2, p1), KeyError)
        p1 = dict(pa1="r", pa2=2, d1=dict(e='e', i=0))
        p2 = dict(pa1="r", pa2=2, d1=dict(e='e', i=0))
        self.assertTrue(SkBase.compare_params(p1, p2))
        p2['d1']['i'] = 3
        self.assertFalse(SkBase.compare_params(p1, p2))
        p2['d1']['i2'] = 3
        self.assertRaise(lambda: SkBase.compare_params(
            p1, p2), ValueError, "Values for key")

    def test_sklearn_compare_object(self):
        p1 = SkBase(pa1="r", pa2=2)
        p2 = SkBase(pa1="r", pa2=2, pa3=4)
        self.assertRaise(lambda: p1.test_equality(p2), KeyError)
        self.assertRaise(lambda: p2.test_equality(p1), KeyError)

        p1 = SkBase(pa1="r", pa2=2, d1=dict(e='e', i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=dict(e='e', i=0))
        self.assertTrue(p1.test_equality(p2))
        p2 = SkBase(pa1="r", pa2=2, d1=dict(e='e', i=3))
        self.assertFalse(p1.test_equality(p2))
        p2 = SkBase(pa1="r", pa2=2, d1=dict(e='e', i=3, i2=4))
        self.assertRaise(lambda: p1.test_equality(p2), ValueError)

        p1 = SkBase(pa1="r", pa2=2, d1=SkBase(e='e', i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=SkBase(e='e', i=0))
        self.assertTrue(p1.test_equality(p2))

        p1 = SkBase(pa1="r", pa2=2, d1=SkBase(e='e', i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=SkBase(e='ef', i=0))
        self.assertRaise(lambda: p1.test_equality(p2), ValueError)

        p1 = SkBase(pa1="r", pa2=2, d1=SkBase(e='e', i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=SkBase(e='e', i=0, i2=4))
        self.assertRaise(lambda: p1.test_equality(p2), KeyError)

        p1 = SkBase(pa1="r", pa2=2, d1=[SkBase(e='e', i=0)])
        p2 = SkBase(pa1="r", pa2=2, d1=[SkBase(e='e', i=0, i2=4)])
        self.assertRaise(lambda: p1.test_equality(p2), KeyError)

        p1 = SkBase(pa1="r", pa2=2, d1=[SkBase(e='e', i=0)])
        p2 = SkBase(pa1="r", pa2=2, d1=[SkBase(e='e', i=0)])
        self.assertTrue(p1.test_equality(p2))

        p1 = SkBase(pa1="r", pa2=2, d1=[
                    SkBase(e='e', i=0), SkBase(e='e', i=0)])
        p2 = SkBase(pa1="r", pa2=2, d1=[SkBase(e='e', i=0)])
        self.assertRaise(lambda: p1.test_equality(p2), ValueError)


if __name__ == "__main__":
    unittest.main()
