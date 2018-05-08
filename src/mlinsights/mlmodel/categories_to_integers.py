"""
@file
@brief Implements a transformation which can be put in a pipeline to transform categories in
integers.
"""

import numpy
import pandas
import textwrap

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
        :tag: Machine Learning

        Example which transforms text into integers:

        .. runpython::
            :showcode:

            import pandas
            from ensae_teaching_cs.ml import CategoriesToIntegers
            df = pandas.DataFrame( [{"cat": "a"}, {"cat": "b"}] )
            trans = CategoriesToIntegers()
            trans.fit(df)
            newdf = trans.transform(df)
            print(newdf)
    """

    def __init__(self, columns=None, remove=None, skip_errors=False, single=False, fLOG=None):
        """
        @param      columns         specify a columns selection
        @param      remove          modalities to remove
        @param      skip_errors     skip when a new categories appear (no 1)
        @param      single          use a single column per category, do not multiply them for each value
        @param      fLOG            logging function

        The logging function displays a message when a new dense and big matrix
        is created when it should be sparse. A sparse matrix should be allocated instead.
        """
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self._p_columns = columns if isinstance(
            columns, list) or columns is None else [columns]
        self._p_skip_errors = skip_errors
        self._p_remove = remove
        self._p_single = single
        self.fLOG = fLOG

    def __repr__(self):
        """
        usual
        """
        s1 = ', '.join("'{0}'".format(_)
                       for _ in self._p_columns) if self._p_columns else None
        s2 = ', '.join("'{0}'".format(_)
                       for _ in self._p_remove) if self._p_remove else None
        s = "DictVectorizerTransformer(columns=[{0}], remove=[{1}], skip_errors={2})".format(
            s1, s2, self._p_skip_errors)
        return "\n".join(textwrap.wrap(s))

    def __str__(self):
        """
        usual
        """
        return self.__repr__()

    def fit(self, X, y=None, **fit_params):
        """
        Makes the list of all categories in input *X*.
        *X* must be a dataframe.

        Parameters
        ----------
        X : iterable
            Training data

        y : iterable, default=None
            Training targets.

        Returns
        -------
        self
        """
        if not isinstance(X, pandas.DataFrame):
            raise TypeError(
                "this transformer only accept Dataframes, not {0}".format(type(X)))
        if self._p_columns:
            columns = self._p_columns
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
                raise ValueError(
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
            if self._p_remove:
                sch = [d for d in sch if d[1] not in self._p_remove]
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

        Parameters
        ----------
        X : iterable
            Training data

        y : iterable, default=None
            Training targets.

        Returns
        -------
        DataFrame, *X* with categories.
        """
        if not isinstance(X, pandas.DataFrame):
            raise TypeError(
                "X is not a dataframe: {0}".format(type(X)))

        if self._p_single:
            b = not self._p_skip_errors

            def transform(v, vec):
                if v in vec:
                    return vec[v]
                elif v is None:
                    return numpy.nan
                elif isinstance(v, float) and numpy.isnan(v):
                    return numpy.nan
                elif not self._p_skip_errors:
                    lv = list(sorted(vec))
                    if len(lv) > 20:
                        lv = lv[:20]
                        lv.append("...")
                    raise ValueError("unable to find category value '{0}': '{1}' type(v)={3} among\n{2}".format(
                        k, v, "\n".join(lv), type(v)))
                else:
                    return numpy.nan

            sch, pos, new_vector = self._schema
            X = X.copy()
            for c in self._fit_columns:
                X[c] = X[c].apply(lambda v: transform(v, new_vector[c]))
            return X
        else:
            dfcat = X[self._fit_columns]
            dfnum = X[[c for c in X.columns if c not in self._fit_columns]]
            sch, pos, new_vector = self._schema
            vec = new_vector

            new_size = X.shape[0] * len(sch)
            if new_size >= 2e30 and self.fLOG:
                self.fLOG("Allocating {0} floats.".format(new_size))
            res = numpy.zeros((X.shape[0], len(sch)))
            res.fill(numpy.nan)
            b = not self._p_skip_errors

            for i, row in enumerate(dfcat.to_dict("records")):
                for k, v in row.items():
                    if v is None or (isinstance(v, float) and numpy.isnan(v)):
                        # missing values
                        continue
                    if v not in vec[k]:
                        if b:
                            lv = list(sorted(vec[k]))
                            if len(lv) > 20:
                                lv = lv[:20]
                                lv.append("...")
                            raise ValueError("unable to find category value '{0}': '{1}' type(v)={3} among\n{2}".format(
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

        Parameters
        ----------
        X : iterable
            Training data

        y : iterable, default=None
            Training targets.

        Returns
        -------
        Dataframe, *X* with categories.
        """
        return self.fit(X, y=y, **fit_params).transform(X, y)
