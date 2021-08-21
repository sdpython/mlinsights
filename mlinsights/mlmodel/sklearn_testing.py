"""
@file
@brief Helpers to test a model which follows :epkg:`scikit-learn` API.
"""
import copy
import pickle
import pprint
from unittest import TestCase
from io import BytesIO
from numpy import ndarray
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def train_test_split_with_none(X, y=None, sample_weight=None, random_state=0):
    """
    Splits into train and test data even if they are None.

    @param      X               X
    @param      y               y
    @param      sample_weight   sample weight
    @param      random_state    random state
    @return                     similar to :epkg:`scikit-learn:model_selection:train_test_split`.
    """
    not_none = [_ for _ in [X, y, sample_weight] if _ is not None]
    res = train_test_split(*not_none)
    inc = len(not_none)
    trains = []
    tests = []
    for i in range(inc):
        trains.append(res[i * 2])
        tests.append(res[i * 2 + 1])
    while len(trains) < 3:
        trains.append(None)
        tests.append(None)
    X_train, y_train, w_train = trains
    X_test, y_test, w_test = tests
    return X_train, y_train, w_train, X_test, y_test, w_test


def test_sklearn_pickle(fct_model, X, y=None, sample_weight=None, **kwargs):
    """
    Creates a model, fit, predict and check the prediction
    are similar after the model was pickled, unpickled.

    @param      fct_model       function which creates the model
    @param      X               X
    @param      y               y
    @param      sample_weight   sample weight
    @param      kwargs          additional parameters for :epkg:`numpy:testing:assert_almost_equal`
    @return                     model, unpickled model

    :raises:
        AssertionError
    """
    X_train, y_train, w_train, X_test, _, __ = train_test_split_with_none(
        X, y, sample_weight)
    model = fct_model()
    if y_train is None and w_train is None:
        model.fit(X_train)
    else:
        try:
            model.fit(X_train, y_train, w_train)
        except TypeError:
            # Do not accept weights?
            model.fit(X_train, y_train)
    if hasattr(model, 'predict'):
        pred1 = model.predict(X_test)
    else:
        pred1 = model.transform(X_test)

    st = BytesIO()
    pickle.dump(model, st)
    data = BytesIO(st.getvalue())
    model2 = pickle.load(data)
    if hasattr(model2, 'predict'):
        pred2 = model2.predict(X_test)
    else:
        pred2 = model2.transform(X_test)
    if isinstance(pred1, ndarray):
        assert_almost_equal(pred1, pred2, **kwargs)
    else:
        assert_frame_equal(pred1, pred2, **kwargs)
    return model, model2


def _get_test_instance():
    try:
        from pyquickhelper.pycode import ExtTestCase  # pylint: disable=C0415
        cls = ExtTestCase
    except ImportError:  # pragma: no cover

        class _ExtTestCase(TestCase):
            "simple test classe with a more methods"

            def assertIsInstance(self, inst, cltype):
                "checks that one instance is from one type"
                if not isinstance(inst, cltype):
                    raise AssertionError(
                        "Unexpected type {} != {}.".format(
                            type(inst), cltype))

        cls = _ExtTestCase
    return cls()


def test_sklearn_clone(fct_model, ext=None, copy_fitted=False):
    """
    Tests that a cloned model is similar to the original one.

    @param      fct_model       function which creates the model
    @param      ext             unit test class instance
    @param      copy_fitted     copy fitted parameters as well
    @return                     model, cloned model

    :raises:
        AssertionError
    """
    conv = fct_model()
    p1 = conv.get_params(deep=True)
    if copy_fitted:
        cloned = clone_with_fitted_parameters(conv)
    else:
        cloned = clone(conv)
    p2 = cloned.get_params(deep=True)
    if ext is None:
        ext = _get_test_instance()
    try:
        ext.assertEqual(set(p1), set(p2))
    except AssertionError as e:  # pragma no cover
        p1 = pprint.pformat(p1)
        p2 = pprint.pformat(p2)
        raise AssertionError(
            "Differences between\n----\n{0}\n----\n{1}".format(p1, p2)) from e

    for k in sorted(p1):
        if isinstance(p1[k], BaseEstimator) and isinstance(p2[k], BaseEstimator):
            if copy_fitted:
                assert_estimator_equal(p1[k], p2[k])
        elif isinstance(p1[k], list) and isinstance(p2[k], list):
            _assert_list_equal(p1[k], p2[k], ext)
        else:
            try:
                ext.assertEqual(p1[k], p2[k])
            except AssertionError:  # pragma no cover
                raise AssertionError(  # pylint: disable=W0707
                    "Difference for key '{0}'\n==1 {1}\n==2 {2}".format(
                        k, p1[k], p2[k]))
    return conv, cloned


def _assert_list_equal(l1, l2, ext):
    if len(l1) != len(l2):
        raise AssertionError(  # pragma no cover
            "Lists have different length {0} != {1}".format(len(l1), len(l2)))
    for a, b in zip(l1, l2):
        if isinstance(a, tuple) and isinstance(b, tuple):
            _assert_tuple_equal(a, b, ext)
        else:
            ext.assertEqual(a, b)


def _assert_dict_equal(a, b, ext):
    if not isinstance(a, dict):  # pragma no cover
        raise TypeError('a is not dict but {0}'.format(type(a)))
    if not isinstance(b, dict):  # pragma no cover
        raise TypeError('b is not dict but {0}'.format(type(b)))
    rows = []
    for key in sorted(b):
        if key not in a:
            rows.append("** Added key '{0}' in b".format(key))
        elif isinstance(a[key], BaseEstimator) and isinstance(b[key], BaseEstimator):
            assert_estimator_equal(a[key], b[key], ext)
        else:
            if a[key] != b[key]:
                rows.append(
                    "** Value != for key '{0}': != id({1}) != id({2})\n==1 {3}\n==2 {4}".format(
                        key, id(a[key]), id(b[key]), a[key], b[key]))
    for key in sorted(a):
        if key not in b:
            rows.append("** Removed key '{0}' in a".format(key))
    if len(rows) > 0:
        raise AssertionError(  # pragma: no cover
            "Dictionaries are different\n{0}".format('\n'.join(rows)))


def _assert_tuple_equal(t1, t2, ext):
    if len(t1) != len(t2):  # pragma no cover
        raise AssertionError(
            "Lists have different length {0} != {1}".format(len(t1), len(t2)))
    for a, b in zip(t1, t2):
        if isinstance(a, BaseEstimator) and isinstance(b, BaseEstimator):
            assert_estimator_equal(a, b, ext)
        else:
            ext.assertEqual(a, b)


def assert_estimator_equal(esta, estb, ext=None):
    """
    Checks that two models are equal.

    @param      esta        first estimator
    @param      estb        second estimator
    @param      ext         unit test class

    The function raises an exception if the comparison fails.
    """
    if ext is None:
        ext = _get_test_instance()
    ext.assertIsInstance(esta, estb.__class__)
    ext.assertIsInstance(estb, esta.__class__)
    _assert_dict_equal(esta.get_params(), estb.get_params(), ext)
    for att in esta.__dict__:
        if (att.endswith('_') and not att.endswith('__')) or \
                (att.startswith('_') and not att.startswith('__')):
            if not hasattr(estb, att):  # pragma no cover
                raise AssertionError("Missing fitted attribute '{}' class {}\n==1 {}\n==2 {}".format(
                    att, esta.__class__, list(sorted(esta.__dict__)), list(sorted(estb.__dict__))))
            if isinstance(getattr(esta, att), BaseEstimator):
                assert_estimator_equal(
                    getattr(esta, att), getattr(estb, att), ext)
            else:
                ext.assertEqual(getattr(esta, att), getattr(estb, att))
    for att in estb.__dict__:
        if att.endswith('_') and not att.endswith('__'):
            if not hasattr(esta, att):  # pragma no cover
                raise AssertionError("Missing fitted attribute\n==1 {}\n==2 {}".format(
                    list(sorted(esta.__dict__)), list(sorted(estb.__dict__))))


def test_sklearn_grid_search_cv(fct_model, X, y=None, sample_weight=None, **grid_params):
    """
    Creates a model, checks that a grid search works with it.

    @param      fct_model       function which creates the model
    @param      X               X
    @param      y               y
    @param      sample_weight   sample weight
    @param      grid_params     parameter to use to run the grid search.
    @return                     dictionary with results

    :raises:
        AssertionError
    """
    X_train, y_train, w_train, X_test, y_test, w_test = train_test_split_with_none(
        X, y, sample_weight)
    model = fct_model()
    pipe = make_pipeline(model)
    name = model.__class__.__name__.lower()
    parameters = {name + "__" + k: v for k, v in grid_params.items()}
    if len(parameters) == 0:
        raise ValueError(
            "Some parameters must be tested when running grid search.")
    clf = GridSearchCV(pipe, parameters)
    if y_train is None and w_train is None:
        clf.fit(X_train)
    elif w_train is None:
        clf.fit(X_train, y_train)  # pylint: disable=E1121
    else:
        clf.fit(X_train, y_train, w_train)  # pylint: disable=E1121
    score = clf.score(X_test, y_test)
    ext = _get_test_instance()
    ext.assertIsInstance(score, float)
    return dict(model=clf, X_train=X_train, y_train=y_train, w_train=w_train,
                X_test=X_test, y_test=y_test, w_test=w_test, score=score)


def clone_with_fitted_parameters(est):
    """
    Clones an estimator with the fitted results.

    @param      est     estimator
    @return             cloned object
    """
    def adjust(obj1, obj2):
        if isinstance(obj1, list) and isinstance(obj2, list):
            for a, b in zip(obj1, obj2):
                adjust(a, b)
        elif isinstance(obj1, tuple) and isinstance(obj2, tuple):
            for a, b in zip(obj1, obj2):
                adjust(a, b)
        elif isinstance(obj1, dict) and isinstance(obj2, dict):
            for a, b in zip(obj1, obj2):
                adjust(obj1[a], obj2[b])
        elif isinstance(obj1, BaseEstimator) and isinstance(obj2, BaseEstimator):
            for k in obj1.__dict__:
                if hasattr(obj2, k):
                    v1 = getattr(obj1, k)
                    if callable(v1):
                        raise RuntimeError(  # pragma: no cover
                            "Cannot migrate trained parameters for {}.".format(obj1))
                    elif isinstance(v1, BaseEstimator):
                        v1 = getattr(obj1, k)
                        setattr(obj2, k, clone_with_fitted_parameters(v1))
                    else:
                        adjust(getattr(obj1, k), getattr(obj2, k))
                elif (k.endswith('_') and not k.endswith('__')) or \
                     (k.startswith('_') and not k.startswith('__')):
                    v1 = getattr(obj1, k)
                    setattr(obj2, k, clone_with_fitted_parameters(v1))
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Cloned object is missing '{0}' in {1}.".format(k, obj2))

    if isinstance(est, BaseEstimator):
        cloned = clone(est)
        adjust(est, cloned)
        res = cloned
    elif isinstance(est, list):
        res = list(clone_with_fitted_parameters(o) for o in est)
    elif isinstance(est, tuple):
        res = tuple(clone_with_fitted_parameters(o) for o in est)
    elif isinstance(est, dict):
        res = {k: clone_with_fitted_parameters(v) for k, v in est.items()}
    else:
        res = copy.deepcopy(est)
    return res
