import unittest
import numpy
from mlinsights.ext_test_case import ExtTestCase
from mlinsights.mlmodel import (
    FunctionReciprocalTransformer,
    PermutationReciprocalTransformer,
)


class TestSklearnTransformInv(ExtTestCase):
    def test_sklearn_transform_inv_log(self):
        X = numpy.array(
            [[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]], dtype=float
        )
        Y = numpy.array([0, 1, 0, 1], dtype=float) + 1
        tr = FunctionReciprocalTransformer("log")
        tr.fit()
        x1, y1 = tr.transform(X, Y)
        self.assertEqualArray(X, x1)
        self.assertEqualArray(numpy.log(Y), y1)
        inv = tr.get_fct_inv()
        x2, y2 = inv.transform(x1, y1)
        self.assertEqualArray(X, x2)
        self.assertEqualArray(Y, y2)

    def test_sklearn_transform_inv_log1(self):
        X = numpy.array(
            [[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]], dtype=float
        )
        Y = numpy.array([0, 1, 0, 1], dtype=float) + 1
        tr = FunctionReciprocalTransformer("log(1+x)")
        tr.fit()
        x1, y1 = tr.transform(X, Y)
        self.assertEqualArray(X, x1)
        self.assertEqualArray(numpy.log(Y + 1), y1)
        inv = tr.get_fct_inv()
        x2, y2 = inv.transform(x1, y1)
        self.assertEqualArray(X, x2)
        self.assertEqualArray(Y, y2, atol=1e-10)

    def test_sklearn_transform_inv_err(self):
        self.assertRaise(
            lambda: FunctionReciprocalTransformer("log(1+x)***"), ValueError
        )

    def test_sklearn_transform_inv_sqrt(self):
        X = numpy.array(
            [[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]], dtype=float
        )
        Y = numpy.array([0, 1, 0, 1], dtype=float) + 1
        tr = FunctionReciprocalTransformer(lambda x: x * x, numpy.sqrt)
        tr.fit()
        x1, y1 = tr.transform(X, Y)
        self.assertEqualArray(X, x1)
        self.assertEqualArray(Y * Y, y1)
        inv = tr.get_fct_inv()
        x2, y2 = inv.transform(x1, y1)
        self.assertEqualArray(X, x2)
        self.assertEqualArray(Y, y2)

    def test_permutation_reciprocal_transformer(self):
        X = numpy.array(
            [[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]], dtype=float
        )
        Y = numpy.array([0, 1, 0, 1], dtype=int) + 1
        p = PermutationReciprocalTransformer(0)
        p.fit(X, Y)
        self.assertTrue(hasattr(p, "permutation_"))
        self.assertIsInstance(p.permutation_, dict)
        self.assertEqual(len(p.permutation_), 2)
        _, y1 = p.transform(X, Y)
        self.assertEqual(Y.shape, y1.shape)

        inv = p.get_fct_inv()
        _, y2 = inv.transform(X, y1)
        self.assertEqual(Y.shape, y2.shape)
        self.assertEqualArray(Y, y2)

    def test_permutation_reciprocal_transformer_nan(self):
        X = numpy.array(
            [[0.1, 0.2], [-0.2, -0.3], [0.2, 0.35], [-0.2, -0.36]], dtype=float
        )
        Y = numpy.array([0, 1, 0, numpy.nan], dtype=float) + 1
        p = PermutationReciprocalTransformer(0)
        p.fit(X, Y)
        self.assertTrue(hasattr(p, "permutation_"))
        self.assertIsInstance(p.permutation_, dict)
        self.assertEqual(len(p.permutation_), 2)
        _, y1 = p.transform(X, Y)
        self.assertEqual(Y.shape, y1.shape)

        inv = p.get_fct_inv()
        _, y2 = inv.transform(X, y1)
        self.assertEqual(Y.shape, y2.shape)
        self.assertEqualArray(Y, y2)
        self.assertTrue(numpy.isnan(y1[-1]))
        self.assertTrue(numpy.isnan(y2[-1]))


if __name__ == "__main__":
    unittest.main()
