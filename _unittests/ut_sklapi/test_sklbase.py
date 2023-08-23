import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlinsights.sklapi.sklearn_base import SkBase
from mlinsights.sklapi.sklearn_base_learner import SkBaseLearner
from mlinsights.sklapi.sklearn_base_regressor import SkBaseRegressor
from mlinsights.sklapi.sklearn_base_classifier import SkBaseClassifier
from mlinsights.sklapi.sklearn_base_transform import SkBaseTransform


class TestSklearnBase(ExtTestCase):
    def test_sklearn_base_parameters(self):
        sk = SkBase(pa1="r", pa2=2)
        p = sk.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk)
        self.assertEqual(r, "SkBase(pa1='r', pa2=2)")
        self.assertTrue(sk.test_equality(sk))
        sk.set_params(r=3)

    def test_sklearn_equality(self):
        sk1 = SkBaseLearner(pa1="r", pa2=2)
        p = sk1.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk1)
        self.assertEqual(r, "SkBaseLearner(pa1='r', pa2=2)")
        sk2 = SkBase(pa1="r", pa2=2)
        self.assertFalse(sk1.test_equality(sk2))

    def test_sklearn_equality_reg(self):
        sk1 = SkBaseRegressor(pa1="r", pa2=2)
        p = sk1.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk1)
        self.assertEqual(r, "SkBaseRegressor(pa1='r', pa2=2)")
        sk2 = SkBase(pa1="r", pa2=2)
        self.assertFalse(sk1.test_equality(sk2))
        x = numpy.array([[0, 1]], dtype=numpy.float64)
        y = numpy.array([0], dtype=numpy.float64)
        self.assertRaise(lambda: sk1.score(x, y), NotImplementedError)

    def test_sklearn_equality_cls(self):
        sk1 = SkBaseClassifier(pa1="r", pa2=2)
        p = sk1.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk1)
        self.assertEqual(r, "SkBaseClassifier(pa1='r', pa2=2)")
        sk2 = SkBase(pa1="r", pa2=2)
        self.assertFalse(sk1.test_equality(sk2))
        x = numpy.array([[0, 1]], dtype=numpy.float64)
        y = numpy.array([0], dtype=numpy.float64)
        self.assertRaise(lambda: sk1.score(x, y), NotImplementedError)
        self.assertRaise(lambda: sk1.predict_proba(x), NotImplementedError)

    def test_sklearn_equality_tr(self):
        sk1 = SkBaseTransform(pa1="r", pa2=2)
        p = sk1.get_params()
        self.assertEqual(p, dict(pa1="r", pa2=2))
        r = repr(sk1)
        self.assertEqual(r, "SkBaseTransform(pa1='r', pa2=2)")
        sk2 = SkBase(pa1="r", pa2=2)
        self.assertFalse(sk1.test_equality(sk2))
        x = numpy.array([[0, 1]], dtype=numpy.float64)
        self.assertRaise(lambda: sk1.transform(x), NotImplementedError)
        self.assertRaise(lambda: sk1.fit(x), NotImplementedError)

    def test_sklearn_compare(self):
        p1 = dict(pa1="r", pa2=2)
        p2 = dict(pa1="r", pa2=2, pa3=4)
        self.assertRaise(lambda: SkBase.compare_params(p1, p2), KeyError)
        self.assertRaise(lambda: SkBase.compare_params(p2, p1), KeyError)
        p1 = dict(pa1="r", pa2=2, d1=dict(e="e", i=0))
        p2 = dict(pa1="r", pa2=2, d1=dict(e="e", i=0))
        self.assertTrue(SkBase.compare_params(p1, p2))
        p2["d1"]["i"] = 3
        self.assertFalse(SkBase.compare_params(p1, p2))
        p2["d1"]["i2"] = 3
        self.assertRaise(
            lambda: SkBase.compare_params(p1, p2), ValueError, "Values for key"
        )

    def test_sklearn_compare_object(self):
        p1 = SkBase(pa1="r", pa2=2)
        p2 = SkBase(pa1="r", pa2=2, pa3=4)
        self.assertRaise(lambda: p1.test_equality(p2), KeyError)
        self.assertRaise(lambda: p2.test_equality(p1), KeyError)

        p1 = SkBase(pa1="r", pa2=2, d1=dict(e="e", i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=dict(e="e", i=0))
        self.assertTrue(p1.test_equality(p2))
        p2 = SkBase(pa1="r", pa2=2, d1=dict(e="e", i=3))
        self.assertFalse(p1.test_equality(p2))
        p2 = SkBase(pa1="r", pa2=2, d1=dict(e="e", i=3, i2=4))
        self.assertRaise(lambda: p1.test_equality(p2), ValueError)

        p1 = SkBase(pa1="r", pa2=2, d1=SkBase(e="e", i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=SkBase(e="e", i=0))
        self.assertTrue(p1.test_equality(p2))

        p1 = SkBase(pa1="r", pa2=2, d1=SkBase(e="e", i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=SkBase(e="ef", i=0))
        self.assertRaise(lambda: p1.test_equality(p2), ValueError)

        p1 = SkBase(pa1="r", pa2=2, d1=SkBase(e="e", i=0))
        p2 = SkBase(pa1="r", pa2=2, d1=SkBase(e="e", i=0, i2=4))
        self.assertRaise(lambda: p1.test_equality(p2), KeyError)

        p1 = SkBase(pa1="r", pa2=2, d1=[SkBase(e="e", i=0)])
        p2 = SkBase(pa1="r", pa2=2, d1=[SkBase(e="e", i=0, i2=4)])
        self.assertRaise(lambda: p1.test_equality(p2), KeyError)

        p1 = SkBase(pa1="r", pa2=2, d1=[SkBase(e="e", i=0)])
        p2 = SkBase(pa1="r", pa2=2, d1=[SkBase(e="e", i=0)])
        self.assertTrue(p1.test_equality(p2))

        p1 = SkBase(pa1="r", pa2=2, d1=[SkBase(e="e", i=0), SkBase(e="e", i=0)])
        p2 = SkBase(pa1="r", pa2=2, d1=[SkBase(e="e", i=0)])
        self.assertRaise(lambda: p1.test_equality(p2), ValueError)


if __name__ == "__main__":
    unittest.main()
