"""
@file
@brief Combines a *k-means* followed by a predictor.
"""
import textwrap
import inspect
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class ClassifierAfterKMeans(BaseEstimator, ClassifierMixin):
    """
    Applies a *k-means* (see :epkg:`sklearn:cluster:KMeans`)
    for each class, then adds the distance to each cluster
    as a feature for a classifier.
    See notebook :ref:`logisticregressionclusteringrst`.
    """

    def __init__(self, estimator=None, clus=None, **kwargs):
        """
        @param  estimator   :epkg:`sklearn:linear_model:LogisiticRegression`
                            by default
        @param  clus        clustering applied on each class,
                            by default k-means with two classes
        @param  kwargs      sent to :meth:`set_params
                            <mlinsights.mlmodel.classification_kmeans.
                            ClassifierAfterKMeans.set_params>`,
                            see its documentation to understand how to
                            specify parameters
        """
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)
        if estimator is None:
            estimator = LogisticRegression()
        if clus is None:
            clus = KMeans(n_clusters=2)
        self.estimator = estimator
        self.clus = clus
        if not hasattr(clus, "transform"):
            raise AttributeError(  # pragma: no cover
                "clus does not have a transform method.")
        if kwargs:
            self.set_params(**kwargs)

    def fit(self, X, y, sample_weight=None):
        """
        Runs a *k-means* on each class
        then trains a classifier on the
        extended set of features.

        :param X: numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        :param y: numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary
        :param sample_weight: numpy array of shape [n_samples]
            Individual weights for each sample
        :return: self : returns an instance of self.

        Fitting attributes:
        * `labels_`: dictionary of clustering models
        * `clus_`: array of clustering models
        * `estimator_`: trained classifier
        """
        classes = set(y)
        self.labels_ = list(sorted(classes))
        self.clus_ = {}
        sig = inspect.signature(self.clus.fit)
        for cl in classes:
            m = clone(self.clus)
            Xcl = X[y == cl]
            if sample_weight is None or 'sample_weight' not in sig.parameters:
                w = None
                m.fit(Xcl)
            else:
                w = sample_weight[y == cl]
                m.fit(Xcl, sample_weight=w)
            self.clus_[cl] = m

        extX = self.transform_features(X)
        self.estimator_ = self.estimator.fit(
            extX, y, sample_weight=sample_weight)
        return self

    def transform_features(self, X):
        """
        Applies all the clustering objects
        on every observations and extends the list of
        features.

        @param      X       features
        @return             extended features
        """
        preds = []
        for _, v in sorted(self.clus_.items()):
            p = v.transform(X)
            preds.append(p)
        return numpy.hstack(preds)

    def predict(self, X):
        """
        Runs the predictions.
        """
        extX = self.transform_features(X)
        return self.estimator.predict(extX)

    def predict_proba(self, X):
        """
        Converts predictions into probabilities.
        """
        extX = self.transform_features(X)
        return self.estimator.predict_proba(extX)

    def decision_function(self, X):
        """
        Calls *decision_function*.
        """
        extX = self.transform_features(X)
        return self.estimator.decision_function(extX)

    def get_params(self, deep=True):
        """
        Returns the parameters for both
        the clustering and the classifier.

        @param      deep        unused here
        @return                 dict

        :meth:`set_params <mlinsights.mlmodel.classification_kmeans.
        ClassifierAfterKMeans.set_params>`
        describes the pattern parameters names follow.
        """
        res = {}
        for k, v in self.clus.get_params().items():
            res["c_" + k] = v
        for k, v in self.estimator.get_params().items():
            res["e_" + k] = v
        return res

    def set_params(self, **values):
        """
        Sets the parameters before training.
        Every parameter prefixed by ``'e_'`` is an estimator
        parameter, every parameter prefixed by ``'c_'`` is for
        the :epkg:`sklearn:cluster:KMeans`.

        @param      values      valeurs
        @return                 dict
        """
        pc, pe = {}, {}
        for k, v in values.items():
            if k.startswith('e_'):
                pe[k[2:]] = v
            elif k.startswith('c_'):
                pc[k[2:]] = v
            else:
                raise ValueError(  # pragma: no cover
                    "Unexpected parameter name '{0}'".format(k))
        self.clus.set_params(**pc)
        self.estimator.set_params(**pe)

    def __repr__(self):  # pylint: disable=W0222
        """
        Overloads `repr` as *scikit-learn* now relies
        on the constructor signature.
        """
        el = ', '.join(['%s=%r' % (k, v)
                        for k, v in self.get_params().items()])
        text = "%s(%s)" % (self.__class__.__name__, el)
        lines = textwrap.wrap(text, subsequent_indent='    ')
        return "\n".join(lines)
