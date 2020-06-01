"""
@brief      test log(time=2s)
"""
import unittest
import pickle
from io import BytesIO
import numpy
import scipy.sparse
import pandas
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.loghelper import BufferedPrint
from mlinsights.mlmodel._kmeans_constraint_ import (
    linearize_matrix, _compute_strategy_coefficient,
    _constraint_association_gain)
from mlinsights.mlmodel import ConstraintKMeans


class TestSklearnConstraintKMeans(ExtTestCase):

    def test_mat_lin(self):
        mat = numpy.identity(3)
        lin = linearize_matrix(mat)
        exp = numpy.array([[1., 0., 0.],
                           [0., 0., 1.],
                           [0., 0., 2.],
                           [0., 1., 0.],
                           [1., 1., 1.],
                           [0., 1., 2.],
                           [0., 2., 0.],
                           [0., 2., 1.],
                           [1., 2., 2.]])
        self.assertEqual(exp, lin)

    def test_mat_lin_add(self):
        mat = numpy.identity(3)
        mat2 = numpy.identity(3) * 3
        lin = linearize_matrix(mat, mat2)
        exp = numpy.array([[1., 0., 0., 3.],
                           [0., 0., 1., 0.],
                           [0., 0., 2., 0.],
                           [0., 1., 0., 0.],
                           [1., 1., 1., 3.],
                           [0., 1., 2., 0.],
                           [0., 2., 0., 0.],
                           [0., 2., 1., 0.],
                           [1., 2., 2., 3.]])
        self.assertEqual(exp, lin)

    def test_mat_lin_sparse(self):
        mat = numpy.identity(3)
        mat[0, 2] = 8
        mat[1, 2] = 5
        mat[2, 1] = 7
        mat = scipy.sparse.csr_matrix(mat)
        lin = linearize_matrix(mat)
        exp = numpy.array([[1., 0., 0.],
                           [8., 0., 2.],
                           [1., 1., 1.],
                           [5., 1., 2.],
                           [7., 2., 1.],
                           [1., 2., 2.]])
        self.assertEqual(exp, lin)

    def test_mat_lin_sparse_add(self):
        mat = numpy.identity(3)
        mat[0, 2] = 8
        mat[1, 2] = 5
        mat[2, 1] = 7
        mat2 = numpy.identity(3) * 3
        mat = scipy.sparse.csr_matrix(mat)
        mat2 = scipy.sparse.csr_matrix(mat2)
        lin = linearize_matrix(mat, mat2)
        exp = numpy.array([[1., 0., 0., 3.],
                           [8., 0., 2., 0.],
                           [1., 1., 1., 3.],
                           [5., 1., 2., 0.],
                           [7., 2., 1., 0.],
                           [1., 2., 2., 3.]])
        self.assertEqual(exp, lin)

    def test_mat_lin_sparse2(self):
        mat = numpy.identity(3)
        mat[0, 1] = 8
        mat[1, 1] = 0
        mat[2, 1] = 7
        mat = scipy.sparse.csr_matrix(mat)
        lin = linearize_matrix(mat)
        exp = numpy.array([[1., 0., 0.],
                           [8., 0., 1.],
                           [7., 2., 1.],
                           [1., 2., 2.]])
        self.assertEqual(exp, lin)

    def test_mat_lin_sparse3(self):
        mat = numpy.identity(3)
        mat[0, 1] = 8
        mat[2, 1] = 7
        mat = scipy.sparse.csr_matrix(mat)
        lin = linearize_matrix(mat)
        exp = numpy.array([[1., 0., 0.],
                           [8., 0., 1.],
                           [1., 1., 1.],
                           [7., 2., 1.],
                           [1., 2., 2.]])
        self.assertEqual(exp, lin)

    def test_mat_sort(self):
        mat = numpy.identity(3)
        mat[2, 0] = 0.3
        mat[1, 0] = 0.2
        mat[0, 0] = 0.1
        exp = numpy.array([[0.1, 0., 0.], [0.2, 1., 0.], [0.3, 0., 1.]])
        sort = mat[mat[:, 0].argsort()]
        self.assertEqual(exp, sort)
        mat.sort(axis=0)
        self.assertNotEqual(exp, mat)
        mat.sort(axis=1)
        self.assertNotEqual(exp, mat)

    @ignore_warnings(category=ConvergenceWarning)
    def test_kmeans_constraint(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1], [1, 1]])
        km = ConstraintKMeans(n_clusters=2, verbose=0, strategy='distance',
                              balanced_predictions=True)
        km.fit(mat)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        self.assertEqualFloat(km.inertia_, 0.455)
        if km.labels_[0] == 0:
            self.assertEqual(km.labels_, numpy.array([0, 1, 0, 1]))
            self.assertEqual(km.cluster_centers_, numpy.array(
                [[-0.05, -0.05], [0.6, 0.6]]))
        else:
            self.assertEqual(km.labels_, numpy.array([1, 0, 1, 0]))
            self.assertEqual(km.cluster_centers_, numpy.array(
                [[0.6, 0.6], [-0.05, -0.05]]))
        pred = km.predict(mat)
        if km.labels_[0] == 0:
            self.assertEqual(pred, numpy.array([0, 1, 0, 1]))
        else:
            self.assertEqual(pred, numpy.array([1, 0, 1, 0]))

    def test_kmeans_constraint_constraint(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1], [1, 1]])
        km = ConstraintKMeans(n_clusters=2, verbose=0, strategy='distance',
                              balanced_predictions=True)
        km.fit(mat)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        self.assertEqualFloat(km.inertia_, 0.455)
        if km.labels_[0] == 0:
            self.assertEqual(km.labels_, numpy.array([0, 1, 0, 1]))
            self.assertEqual(km.cluster_centers_, numpy.array(
                [[-0.05, -0.05], [0.6, 0.6]]))
        else:
            self.assertEqual(km.labels_, numpy.array([1, 0, 1, 0]))
            self.assertEqual(km.cluster_centers_, numpy.array(
                [[0.6, 0.6], [-0.05, -0.05]]))
        pred = km.predict(mat)
        if km.labels_[0] == 0:
            self.assertEqual(pred, numpy.array([0, 1, 0, 1]))
        else:
            self.assertEqual(pred, numpy.array([1, 0, 1, 0]))

    @ignore_warnings(category=ConvergenceWarning)
    def test_kmeans_constraint_sparse(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1], [1, 1]])
        mat = scipy.sparse.csr_matrix(mat)
        km = ConstraintKMeans(n_clusters=2, verbose=0, strategy='distance')
        km.fit(mat)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        self.assertEqualFloat(km.inertia_, 0.455)
        if km.labels_[0] == 0:
            self.assertEqual(km.labels_, numpy.array([0, 1, 0, 1]))
            self.assertEqual(km.cluster_centers_, numpy.array(
                [[-0.05, -0.05], [0.6, 0.6]]))
        else:
            self.assertEqual(km.labels_, numpy.array([1, 0, 1, 0]))
            self.assertEqual(km.cluster_centers_, numpy.array(
                [[0.6, 0.6], [-0.05, -0.05]]))
        pred = km.predict(mat)
        if km.labels_[0] == 0:
            self.assertEqual(pred, numpy.array([0, 0, 0, 1]))
        else:
            self.assertEqual(pred, numpy.array([1, 1, 1, 0]))

    def test_kmeans_constraint_pipeline(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        km = ConstraintKMeans(strategy='distance')
        pipe = make_pipeline(km, LogisticRegression())
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = accuracy_score(y_test, pred)
        self.assertGreater(score, 0.8)
        score2 = pipe.score(X_test, y_test)
        self.assertEqual(score, score2)
        rp = repr(km)
        self.assertStartsWith("ConstraintKMeans(", rp)

    def test_kmeans_constraint_grid(self):
        df = pandas.DataFrame(dict(y=[0, 1, 0, 1, 0, 1, 0, 1],
                                   X1=[0.5, 0.6, 0.52, 0.62,
                                       0.5, 0.6, 0.51, 0.61],
                                   X2=[0.5, 0.6, 0.7, 0.5,
                                       1.5, 1.6, 1.7, 1.8]))
        X = df.drop('y', axis=1)
        y = df['y']
        model = make_pipeline(ConstraintKMeans(random_state=0, strategy='distance'),
                              DecisionTreeClassifier())
        res = model.get_params(True)
        self.assertNotEmpty(res)

        parameters = {
            'constraintkmeans__n_clusters': [2, 3, 4],
            'constraintkmeans__balanced_predictions': [False, True],
        }
        clf = GridSearchCV(model, parameters, cv=3)
        clf.fit(X, y)
        pred = clf.predict(X)
        self.assertEqual(pred.shape, (8,))

    def test_kmeans_constraint_pickle(self):
        df = pandas.DataFrame(dict(y=[0, 1, 0, 1, 0, 1, 0, 1],
                                   X1=[0.5, 0.6, 0.52, 0.62,
                                       0.5, 0.6, 0.51, 0.61],
                                   X2=[0.5, 0.6, 0.7, 0.5, 1.5, 1.6, 1.7, 1.8]))
        X = df.drop('y', axis=1)
        y = df['y']
        model = ConstraintKMeans(n_clusters=2, strategy='distance')
        model.fit(X, y)
        pred = model.transform(X)
        st = BytesIO()
        pickle.dump(model, st)
        st = BytesIO(st.getvalue())
        rec = pickle.load(st)
        pred2 = rec.transform(X)
        self.assertEqualArray(pred, pred2)

    def test__compute_sortby_coefficient(self):
        m1 = numpy.array([[1., 2.], [4., 5.]])
        labels = [0, 1]
        res = _compute_strategy_coefficient(m1, 'gain', labels)
        exp = numpy.array([[0, 1.], [-1., 0]])
        self.assertEqualArray(res, exp)

    def test_kmeans_constraint_exc(self):
        self.assertRaise(lambda: ConstraintKMeans(
            n_clusters=2, strategy='r'), ValueError)

    def test_kmeans_constraint_none(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1], [1, 1]])
        km = ConstraintKMeans(n_clusters=2, verbose=0, kmeans0=False,
                              random_state=2, strategy='distance')
        km.fit(mat)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        self.assertEqualFloat(km.inertia_, 0.455)
        self.assertEqual(km.cluster_centers_, numpy.array(
            [[-0.05, -0.05], [0.6, 0.6]]))
        self.assertEqual(km.labels_, numpy.array([0, 1, 0, 1]))
        pred = km.predict(mat)
        self.assertEqual(pred, numpy.array([0, 0, 0, 1]))

    def test_kmeans_constraint_gain(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1], [1, 1]])
        km = ConstraintKMeans(n_clusters=2, verbose=0, kmeans0=False,
                              random_state=1, strategy='gain')
        km.fit(mat)
        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        self.assertEqualFloat(km.inertia_, 0.455)
        self.assertEqual(km.cluster_centers_, numpy.array(
            [[0.6, 0.6], [-0.05, -0.05]]))
        self.assertEqual(km.labels_, numpy.array([1, 0, 1, 0]))
        pred = km.predict(mat)
        self.assertEqual(pred, numpy.array([1, 1, 1, 0]))

    def test_kmeans_constraint_gain3(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1],
                           [1, 1], [1.1, 0.9], [-1.1, 1.]])
        # Choose random_state=2 to get the labels [1 1 0 2 2 0].
        # This configuration can only be modified with a permutation
        # of 3 elements.
        km = ConstraintKMeans(n_clusters=3, verbose=0, kmeans0=False,
                              random_state=1, strategy='gain',
                              balanced_predictions=True)
        km.fit(mat)
        self.assertEqual(km.cluster_centers_.shape, (3, 2))
        lab = km.labels_
        self.assertEqual(lab[1], lab[2])
        self.assertEqual(lab[0], lab[5])
        self.assertEqual(lab[3], lab[4])
        pred = km.predict(mat)
        self.assertEqualArray(pred, lab)

    def test_kmeans_constraint_blobs(self):
        data = make_blobs(n_samples=8, n_features=2, centers=2, cluster_std=1.0,
                          center_box=(-10.0, 0.0), shuffle=True, random_state=0)
        X1 = data[0]
        data = make_blobs(n_samples=4, n_features=2, centers=2, cluster_std=1.0,
                          center_box=(0.0, 10.0), shuffle=True, random_state=0)
        X2 = data[0]
        X = numpy.vstack([X1, X2])
        km = ConstraintKMeans(n_clusters=4, verbose=0, kmeans0=False,
                              random_state=2, strategy='gain',
                              balanced_predictions=True)
        km.fit(X)
        self.assertEqual(km.labels_[-2], km.labels_[-1])
        self.assertIn(km.labels_[-1], {km.labels_[-4], km.labels_[-3]})

    def test_kmeans_contrainst_association_gain_ex(self):
        X = numpy.array([[-4.36782139, -1.39383283],
                         [-2.47828717, -4.75632643],
                         [-3.21132851, -4.42949315],
                         [-3.52850301, -4.21749384],
                         [-4.61508381, -2.43750783],
                         [-2.64430697, -3.82538422],
                         [-3.65929854, -5.40526391],
                         [-3.56177654, -2.99946354],
                         [6.17167733, 6.90310534],
                         [5.92441491, 5.85943033],
                         [6.43822346, 7.00053646],
                         [7.35569303, 6.17461578]])
        centers = numpy.array([[-3.10434484, -4.52679231],
                               [6.47250218, 6.48442198],
                               [-4.4914526, -1.91567033],
                               [-3.56177654, -2.99946354]])
        distances_close = numpy.array([0] * X.shape[0])
        labels = numpy.array([2, 0, 0, 0, 2, 0, 0, 3, 1, 3, 1, 1])
        _constraint_association_gain(numpy.array([0, 0, 0, 0]), numpy.array([0, 0, 0, 0]),
                                     labels, numpy.array(
                                         [0, 0, 0, 0]), distances_close,
                                     centers, X, x_squared_norms=None, limit=3, strategy="gain")

    def test_kmeans_constraint_blobs20(self):
        data = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=1.0,
                          center_box=(-10.0, 0.0), shuffle=True, random_state=0)
        X1 = data[0]
        data = make_blobs(n_samples=10, n_features=2, centers=2, cluster_std=1.0,
                          center_box=(0.0, 10.0), shuffle=True, random_state=0)
        X2 = data[0]
        X = numpy.vstack([X1, X2])
        km = ConstraintKMeans(n_clusters=4, verbose=0, kmeans0=False,
                              random_state=2, strategy='gain',
                              balanced_predictions=True,
                              history=True)
        km.fit(X)
        pred = km.predict(X)
        diff = numpy.abs(km.labels_ - pred).sum()
        self.assertLesser(diff, 6)
        cls = km.cluster_centers_iter_
        self.assertEqual(len(cls.shape), 3)

    def test_kmeans_constraint_weights(self):
        mat = numpy.array([[0, 0], [0.2, 0.2], [-0.1, -0.1], [1, 1]])
        km = ConstraintKMeans(n_clusters=2, verbose=10, kmeans0=False,
                              random_state=1, strategy='weights')
        buf = BufferedPrint()
        km.fit(mat, fLOG=buf.fprint)

        km = ConstraintKMeans(n_clusters=2, verbose=5, kmeans0=False,
                              random_state=1, strategy='weights')
        km.fit(mat, fLOG=buf.fprint)

        self.assertEqual(km.cluster_centers_.shape, (2, 2))
        self.assertLesser(km.inertia_, 4.55)
        self.assertEqual(km.cluster_centers_, numpy.array(
            [[0.6, 0.6], [-0.05, -0.05]]))
        self.assertEqual(km.labels_, numpy.array([1, 0, 1, 0]))
        pred = km.predict(mat)
        self.assertEqual(pred, numpy.array([1, 1, 1, 0]))
        dist = km.transform(mat)
        self.assertEqual(dist.shape, (4, 2))
        score = km.score(mat)
        self.assertEqual(score.shape, (4, ))
        self.assertIn("CKMeans", str(buf))

    def test_kmeans_constraint_weights_bigger(self):
        n_samples = 100
        data = make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=1.0,
                          center_box=(-10.0, 0.0), shuffle=True, random_state=2)
        X1 = data[0]
        data = make_blobs(n_samples=n_samples // 2, n_features=2, centers=2, cluster_std=1.0,
                          center_box=(0.0, 10.0), shuffle=True, random_state=2)
        X2 = data[0]
        X = numpy.vstack([X1, X2])
        km = ConstraintKMeans(n_clusters=4, strategy='weights', history=True)
        km.fit(X)
        cl = km.predict(X)
        self.assertEqual(cl.shape, (X.shape[0], ))
        cls = km.cluster_centers_iter_
        self.assertEqual(len(cls.shape), 3)
        edges = km.cluster_edges()
        self.assertIsInstance(edges, set)
        self.assertEqual(len(edges), 5)
        self.assertIsInstance(list(edges)[0], tuple)


if __name__ == "__main__":
    unittest.main()
