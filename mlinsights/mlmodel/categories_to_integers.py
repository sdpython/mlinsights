"""
@file
@brief Implements a transformation which can be put in a pipeline to transform categories in
integers.
"""
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin


class CategoriesToIntegers(BaseEstimator, TransformerMixin):
    """
    Does something similar to what
    `DictVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`_
    does but in a transformer. The method *fit* retains all categories,
    the method *transform* transforms categories into integers.
    Categories are sorted by columns. If the method *transform* tries to convert
    a categories which was not seen by method *fit*, it can raise an exception
    or ignore it and replace it by zero.

    .. exref::
        :title: DictVectorizer or CategoriesToIntegers
        :tag: sklearn

        Example which transforms text into integers:

        .. runpython::
            :showcode:

            import pandas
            from mlinsights.mlmodel import CategoriesToIntegers
            df = pandas.DataFrame( [{"cat": "a"}, {"cat": "b"}] )
            trans = CategoriesToIntegers()
            trans.fit(df)
            newdf = trans.transform(df)
            print(newdf)
    """

    def __init__(self, columns=None, remove=None, skip_errors=False, single=False):
        """
        @param      columns         specify a columns selection
        @param      remove          modalities to remove
        @param      skip_errors     skip when a new categories appear (no 1)
        @param      single          use a single column per category, do not multiply them for each value

        The logging function displays a message when a new dense and big matrix
        is created when it should be sparse. A sparse matrix should be allocated instead.
        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.columns = columns if isinstance(
            columns, list) or columns is None else [columns]
        self.skip_errors = skip_errors
        self.remove = remove
        self.single = single

    def __str__(self):
        """
        usual
        """
        return self.__repr__()

    def fit(self, X, y=None, **fit_params):
        """
        Makes the list of all categories in input *X*.
        *X* must be a dataframe.

        :param X: iterable
            Training data
        :param y: iterable, default=None
            Training targets.
        :return: self
        """
        if not isinstance(X, pandas.DataFrame):
            raise TypeError(  # pragma: no cover
                "this transformer only accept Dataframes, not {0}".format(type(X)))
        if self.columns:
            columns = self.columns
        else:
            columns = [c for c, d in zip(
                X.columns, X.dtypes) if d in (object,)]

        self._fit_columns = columns
        max_cat = max(len(X) // 2 + 1, 10000)

        self._categories = {}
        for c in columns:
            distinct = set(X[c].dropna())
            nb = len(distinct)
            if nb >= max_cat:
                raise ValueError(  # pragma: no cover
                    "Too many categories ({0}) for one column '{1}' max_cat={2}".format(nb, c, max_cat))
            self._categories[c] = dict((c, i)
                                       for i, c in enumerate(list(sorted(distinct))))
        self._schema = self._build_schema()
        return self

    def _build_schema(self):
        """
        Concatenates all the categories
        given the information stored in *_categories*.

        @return             list of columns, beginning of each
        """
        schema = []
        position = {}
        new_vector = {}
        last = 0
        for c, v in self._categories.items():
            sch = [(_[1], "{0}={1}".format(c, _[1]))
                   for _ in sorted((n, d) for d, n in v.items())]
            if self.remove:
                sch = [d for d in sch if d[1] not in self.remove]
            position[c] = last
            new_vector[c] = {d[0]: i for i, d in enumerate(sch)}
            last += len(sch)
            schema.extend(_[1] for _ in sch)

        return schema, position, new_vector

    def transform(self, X, y=None, **fit_params):
        """
        Transforms categories in numerical features based on the list
        of categories found by method *fit*.
        *X* must be a dataframe. The function does not preserve
        the order of the columns.

        :param X: iterable
            Training data
        :param y: iterable, default=None
            Training targets.
        :return: DataFrame, *X* with categories.
        """
        if not isinstance(X, pandas.DataFrame):
            raise TypeError(  # pragma: no cover
                "X is not a dataframe: {0}".format(type(X)))

        if self.single:
            b = not self.skip_errors

            def transform(v, vec):
                "transform a vector"
                if v in vec:
                    return vec[v]
                if v is None:
                    return numpy.nan
                if isinstance(v, float) and numpy.isnan(v):
                    return numpy.nan
                if not self.skip_errors:
                    lv = list(sorted(vec))
                    if len(lv) > 20:  # pragma: no cover
                        lv = lv[:20]
                        lv.append("...")
                    raise ValueError(  # pragma: no cover
                        "Unable to find category value '{0}' type(v)={2} among\n{1}".format(
                            v, "\n".join(lv), type(v)))
                return numpy.nan

            sch, pos, new_vector = self._schema
            X = X.copy()
            for c in self._fit_columns:
                X[c] = X[c].apply(lambda v, cv=c: transform(v, new_vector[cv]))
            return X
        else:
            dfcat = X[self._fit_columns]
            dfnum = X[[c for c in X.columns if c not in self._fit_columns]]
            sch, pos, new_vector = self._schema
            vec = new_vector

            # new_size = X.shape[0] * len(sch)
            res = numpy.zeros((X.shape[0], len(sch)))
            res.fill(numpy.nan)
            b = not self.skip_errors

            for i, row in enumerate(dfcat.to_dict("records")):
                for k, v in row.items():
                    if v is None or (isinstance(v, float) and numpy.isnan(v)):
                        # missing values
                        continue
                    if v not in vec[k]:
                        if b:
                            lv = list(sorted(vec[k]))
                            if len(lv) > 20:  # pragma: no cover
                                lv = lv[:20]
                                lv.append("...")
                            raise ValueError(  # pragma: no cover
                                "unable to find category value '{0}': '{1}' type(v)={3} among\n{2}".format(
                                    k, v, "\n".join(lv), type(v)))
                    else:
                        p = pos[k] + vec[k][v]
                    res[i, p] = 1.0

            if dfnum.shape[1] > 0:
                newdf = pandas.DataFrame(res, columns=sch, index=dfcat.index)
                allnum = pandas.concat([dfnum, newdf], axis=1)
            else:
                allnum = pandas.DataFrame(res, columns=sch, index=dfcat.index)

            return allnum

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits and transforms categories in numerical features based on the list
        of categories found by method *fit*.
        *X* must be a dataframe. The function does not preserve
        the order of the columns.

        :param X: iterable
            Training data
        :param y: iterable, default=None
            Training targets.
        :return: Dataframe, *X* with categories.
        """
        return self.fit(X, y=y, **fit_params).transform(X, y)
