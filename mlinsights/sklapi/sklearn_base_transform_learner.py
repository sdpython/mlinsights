# -*- coding: utf-8 -*-
"""
@file
@brief Implements a *transform* which converts a *learner* into
a *transform*.
"""
import textwrap
import numpy
from .sklearn_base_transform import SkBaseTransform


class SkBaseTransformLearner(SkBaseTransform):
    """
    A *transform* which hides a *learner*, it converts
    method *predict* into *transform*. This way,
    two learners can be inserted into the same pipeline.
    There is another a,d shorter implementation
    with class @see class TransferTransformer.

    .. exref::
        :title: Use two learners into a same pipeline
        :tag: sklearn
        :lid: ex-pipe2learner

        It is impossible to use two *learners* into a pipeline
        unless we use a class such as @see cl SkBaseTransformLearner
        which disguise a *learner* into a *transform*.

        .. runpython::
            :showcode:
            :warningout: FutureWarning

            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_iris
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.pipeline import make_pipeline
            from mlinsights.sklapi import SkBaseTransformLearner

            data = load_iris()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            try:
                pipe = make_pipeline(LogisticRegression(),
                                     DecisionTreeClassifier())
            except Exception as e:
                print("ERROR:")
                print(e)
                print('.')

            pipe = make_pipeline(SkBaseTransformLearner(LogisticRegression()),
                                 DecisionTreeClassifier())
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            score = accuracy_score(y_test, pred)
            print("pipeline avec deux learners :", score)
    """

    def __init__(self, model=None, method=None, **kwargs):
        """
        @param  model   learner instance
        @param  method  method to call to transform the feature (see below)
        @param  kwargs  parameters

        Options for parameter *method*:

        * ``'predict'``
        * ``'predict_proba'``
        * ``'decision_function'``
        * a function

        If *method is None*, the function tries first
        ``predict_proba`` then ``predict`` until one of them
        is part of the class.
        """
        super().__init__(**kwargs)
        self.model = model
        if model is None:
            raise ValueError("value cannot be None")  # pragma: no cover
        if method is None:
            for name in ['predict_proba', 'predict', 'transform']:
                if hasattr(model.__class__, name):
                    method = name
            if method is None:
                raise ValueError(  # pragma: no cover
                    "Unable to guess a default method for '{0}'".format(repr(model)))
        self.method = method
        self._set_method(method)

    def _set_method(self, method):
        """
        Defines the method to use to convert the features
        into predictions.
        """
        if isinstance(method, str):
            if method == 'predict':
                self.method_ = self.model.predict
            elif method == 'predict_proba':
                self.method_ = self.model.predict_proba
            elif method == 'decision_function':
                self.method_ = self.model.decision_function
            elif method == 'transform':
                self.method_ = self.model.transform
            else:
                raise ValueError(  # pragma: no cover
                    "Unexpected method '{0}'".format(method))
        elif callable(method):
            self.method_ = method
        else:
            raise TypeError(  # pragma: no cover
                "Unable to find the transform method, method={0}".format(method))

    def fit(self, X, y=None, **kwargs):
        """
        Trains a model.

        @param      X               features
        @param      y               targets
        @param      kwargs          additional parameters
        @return                     self
        """
        self.model.fit(X, y=y, **kwargs)
        return self

    def transform(self, X):
        """
        Predictions, output of the embedded learner.

        @param      X   features
        @return         prédictions
        """
        res = self.method_(X)
        if len(res.shape) == 1:
            res = res[:, numpy.newaxis]
        return res

    ##############
    # cloning API
    ##############

    def get_params(self, deep=True):
        """
        Returns the parameters mandatory to clone the class.

        @param      deep        unused here
        @return                 dict
        """
        res = self.P.to_dict()
        res['model'] = self.model
        res['method'] = self.method
        if deep:
            par = self.model.get_params(deep)
            for k, v in par.items():
                res["model__" + k] = v
        return res

    def set_params(self, **values):
        """
        Sets parameters.

        @param      values      parameters
        """
        if 'model' in values:
            self.model = values['model']
            del values['model']
        elif not hasattr(self, 'model') or self.model is None:
            raise KeyError(  # pragma: no cover
                "Missing key '{0}' in [{1}]".format(
                    'model', ', '.join(sorted(values))))
        if 'method' in values:
            self._set_method(values['method'])
            del values['method']
        for k in values:
            if not k.startswith('model__'):
                raise ValueError(  # pragma: no cover
                    "Parameter '{0}' must start with 'model__'.".format(k))
        d = len('model__')
        pars = {k[d:]: v for k, v in values.items()}
        self.model.set_params(**pars)
        if 'method' in values:
            self.method = values['method']
            self._set_method(values['method'])

    #################
    # common methods
    #################

    def __repr__(self):
        """
        usual
        """
        rp = repr(self.model)
        rps = repr(self.P)
        res = "{0}(model={1}, method={2}, {3})".format(
            self.__class__.__name__, rp, self.method, rps)
        return "\n".join(textwrap.wrap(res, subsequent_indent="    "))
