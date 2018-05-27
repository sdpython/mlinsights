"""
@file
@brief Helpers to test a model which follows :epkg:`scikit-learn` API.
"""
import pickle
import pprint
from io import BytesIO
from numpy.testing import assert_almost_equal
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from pyquickhelper.pycode import ExtTestCase


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
    model.fit(X_train, y_train, w_train)
    pred1 = model.predict(X_test)

    st = BytesIO()
    pickle.dump(model, st)
    data = BytesIO(st.getvalue())
    model2 = pickle.load(data)
    pred2 = model2.predict(X_test)
    assert_almost_equal(pred1, pred2, **kwargs)
    return model, model2


def test_sklearn_clone(fct_model):
    """
    Tests that a cloned model is similar to the original one.

    @param      fct_model       function which creates the model
    @return                     model, cloned model

    :raises:
        AssertionError
    """
    conv = fct_model()
    p1 = conv.get_params(deep=True)
    cloned = clone(conv)
    p2 = cloned.get_params(deep=True)
    ext = ExtTestCase()
    try:
        ext.assertEqual(p1, p2)
    except AssertionError as e:
        p1 = pprint.pformat(p1)
        p2 = pprint.pformat(p2)
        raise AssertionError(
            "Differences between\n----\n{0}\n----\n{1}".format(p1, p2)) from e
    return conv, cloned


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
    clf.fit(X_train, y_train, w_train)
    score = clf.score(X_test, y_test)
    ExtTestCase().assertIsInstance(score, float)
    return dict(model=clf, X_train=X_train, y_train=y_train, w_train=w_train,
                X_test=X_test, y_test=y_test, w_test=w_test, score=score)
