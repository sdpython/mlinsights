# -*- coding: utf-8 -*-
"""
@file
@brief Implémente un *transform* qui suit la même API que tout :epkg:`scikit-learn` transform.
"""
import textwrap
import numpy
from .sklearn_base_transform import SkBaseTransform
from .sklearn_base_transform_learner import SkBaseTransformLearner


class SkBaseTransformStacking(SkBaseTransform):
    """
    Un *transform* qui cache plusieurs *learners*, arrangés
    selon la méthode du `stacking <http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/>`_.

    .. exref::
        :title: Stacking de plusieurs learners dans un pipeline scikit-learn.
        :tag: sklearn
        :lid: ex-pipe2learner2

        Ce *transform* assemble les résultats de plusieurs learners.
        Ces features servent d'entrée à un modèle de stacking.

        .. runpython::
            :showcode:
            :warningout: FutureWarning

            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_iris
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.pipeline import make_pipeline
            from mlinsights.sklapi import SkBaseTransformStacking

            data = load_iris()
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            trans = SkBaseTransformStacking([LogisticRegression(),
                                             DecisionTreeClassifier()])
            trans.fit(X_train, y_train)
            pred = trans.transform(X_test)
            print(pred[3:])
    """

    def __init__(self, models=None, method=None, **kwargs):
        """
        @param  models  list of learners
        @param  method  methods or list of methods to call
                        to convert features into prediction
                        (see below)
        @param  kwargs  parameters

        Available options for parameter *method*:

        * ``'predict'``
        * ``'predict_proba'``
        * ``'decision_function'``
        * a function

        If *method is None*, the default value is first
        ``predict_proba`` it it exists then ``predict``.
        """
        super().__init__(**kwargs)
        if models is None:
            raise ValueError("models cannot be None")  # pragma: no cover
        if not isinstance(models, list):
            raise TypeError(  # pragma: no cover
                "models must be a list not {0}".format(type(models)))
        if method is None:
            method = 'predict'
        if not isinstance(method, str):
            raise TypeError(  # pragma: no cover
                "Method must be a string not {0}".format(type(method)))
        self.method = method
        if isinstance(method, list):
            if len(method) != len(models):
                raise ValueError(  # pragma: no cover
                    "models and methods must have the same length: {0} != {1}".format(
                        len(models), len(method)))
        else:
            method = [method for m in models]

        def convert2transform(c, new_learners):
            "converting function into a transform"
            m, me = c
            if isinstance(m, SkBaseTransformLearner):
                if me == m.method:
                    return m
                res = SkBaseTransformLearner(m.model, me)
                new_learners.append(res)
                return res
            if hasattr(m, 'transform'):
                return m
            res = SkBaseTransformLearner(m, me)
            new_learners.append(res)
            return res

        new_learners = []
        res = list(map(lambda c: convert2transform(
            c, new_learners), zip(models, method)))
        if len(new_learners) == 0:
            # We need to do that to avoid creating new objects
            # when it is not necessary. This behavior is not
            # supported anymore by scikit-learn.
            # See sklearn.base.py
            self.models = models
        else:
            self.models = res

    def fit(self, X, y=None, **kwargs):
        """
        Trains a model.

        @param      X               features
        @param      y               targets
        @param      kwargs          additional parameters
        @return                     self
        """
        for m in self.models:
            m.fit(X, y=y, **kwargs)
        return self

    def transform(self, X):
        """
        Calls the learners predictions to convert
        the features.

        @param      X   features
        @return         prédictions
        """
        Xs = [m.transform(X) for m in self.models]
        return numpy.hstack(Xs)

    ##############
    # cloning API
    ##############

    def get_params(self, deep=True):
        """
        Returns the parameters which define the object.
        It follows :epkg:`scikit-learn` API.

        @param      deep        unused here
        @return                 dict
        """
        res = self.P.to_dict()
        res['models'] = self.models
        res['method'] = self.method
        if deep:
            for i, m in enumerate(self.models):
                par = m.get_params(deep)
                for k, v in par.items():
                    res["models_{0}__".format(i) + k] = v
        return res

    def set_params(self, **values):
        """
        Sets the parameters.

        @param      params      parameters
        """
        if 'models' in values:
            self.models = values['models']
            del values['models']
        if 'method' in values:
            self.method = values['method']
            del values['method']
        for k, v in values.items():
            if not k.startswith('models_'):
                raise ValueError(  # pragma: no cover
                    "Parameter '{0}' must start with 'models_'.".format(k))
        d = len('models_')
        pars = [{} for m in self.models]
        for k, v in values.items():
            si = k[d:].split('__', 1)
            i = int(si[0])
            pars[i][k[d + 1 + len(si):]] = v
        for p, m in zip(pars, self.models):
            if p:
                m.set_params(**p)

    #################
    # common methods
    #################

    def __repr__(self):
        """
        usual
        """
        rps = repr(self.P)
        res = "{0}([{1}], [{2}], {3})".format(
            self.__class__.__name__,
            ", ".join(repr(m.model if hasattr(m, 'model') else m)
                      for m in self.models),
            ", ".join(repr(m.method if hasattr(m, 'method') else None) for m in self.models), rps)
        return "\n".join(textwrap.wrap(res, subsequent_indent="    "))
