"""
@file
@brief Implements a transform which modifies the target
and applies the reverse transformation on the target.
"""
import numpy
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from .sklearn_transform_inv import BaseReciprocalTransformer


class FunctionReciprocalTransformer(BaseReciprocalTransformer):
    """
    The transform is used to apply a function on a the target,
    predict, then transform the target back before scoring.
    The transforms implements a series of predefined functions:

    .. runpython::
        :showcode:

        import pprint
        from mlinsights.mlmodel.sklearn_transform_inv_fct import FunctionReciprocalTransformer
        pprint.pprint(FunctionReciprocalTransformer.available_fcts())
    """

    @staticmethod
    def available_fcts():
        """
        Returns the list of predefined functions.
        """
        return {
            'log': (numpy.log, 'exp'),
            'exp': (numpy.exp, 'log'),
            'log(1+x)': (lambda x: numpy.log(x + 1), 'exp(x)-1'),
            'log1p': (numpy.log1p, 'expm1'),
            'exp(x)-1': (lambda x: numpy.exp(x) - 1, 'log'),
            'expm1': (numpy.expm1, 'log1p'),
        }

    def __init__(self, fct, fct_inv=None):
        """
        @param      fct         function name of numerical function
        @param      fct_inv     optional if *fct* is a function name,
                                reciprocal function otherwise
        """
        BaseReciprocalTransformer.__init__(self)
        if isinstance(fct, str):
            if fct_inv is not None:
                raise ValueError(  # pragma: no cover
                    "If fct is a function name, fct_inv must not be specified.")
            opts = self.__class__.available_fcts()
            if fct not in opts:
                raise ValueError(  # pragma: no cover
                    "Unknown fct '{}', it should in {}.".format(
                        fct, list(sorted(opts))))
        else:
            if fct_inv is None:
                raise ValueError(
                    "If fct is callable, fct_inv must be specified.")
        self.fct = fct
        self.fct_inv = fct_inv

    def fit(self, X=None, y=None, sample_weight=None):
        """
        Just defines *fct* and *fct_inv*.
        """
        if callable(self.fct):
            self.fct_ = self.fct
            self.fct_inv_ = self.fct_inv
        else:
            opts = self.__class__.available_fcts()
            self.fct_, self.fct_inv_ = opts[self.fct]
        return self

    def get_fct_inv(self):
        """
        Returns a trained transform which reverse the target
        after a predictor.
        """
        if isinstance(self.fct_inv_, str):
            res = FunctionReciprocalTransformer(self.fct_inv_)
        else:
            res = FunctionReciprocalTransformer(self.fct_inv_, self.fct_)
        return res.fit()

    def transform(self, X, y):
        """
        Transforms *X* and *y*.
        Returns transformed *X* and *y*.
        If *y* is None, the returned value for *y*
        is None as well.
        """
        if y is None:
            return X, None
        return X, self.fct_(y)


class PermutationReciprocalTransformer(BaseReciprocalTransformer):
    """
    The transform is used to permute targets,
    predict, then permute the target back before scoring.
    nan values remain nan values. Once fitted, the transform
    has attribute ``permutation_`` which keeps
    track of the permutation to apply.
    """

    def __init__(self, random_state=None, closest=False):
        """
        @param      random_state    random state
        @param      closest         if True, finds the closest permuted element
        """
        BaseReciprocalTransformer.__init__(self)
        self.random_state = random_state
        self.closest = closest

    def fit(self, X=None, y=None, sample_weight=None):
        """
        Defines a random permutation over the targets.
        """
        if y is None:
            raise RuntimeError(  # pragma: no cover
                "targets cannot be empty.")
        num = numpy.issubdtype(y.dtype, numpy.floating)
        perm = {}
        for u in y.ravel():
            if num and numpy.isnan(u):
                continue
            if u in perm:
                continue
            perm[u] = len(perm)

        lin = numpy.arange(len(perm))
        if self.random_state is None:
            lin = numpy.random.permutation(lin)
        else:
            rs = numpy.random.RandomState(  # pylint: disable=E1101
                self.random_state)  # pylint: disable=E1101
            lin = rs.permutation(lin)

        for u in perm:
            perm[u] = lin[perm[u]]
        self.permutation_ = perm

    def _check_is_fitted(self):
        if not hasattr(self, 'permutation_'):
            raise NotFittedError(  # pragma: no cover
                "This instance {} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method.".format(
                    type(self)))

    def get_fct_inv(self):
        """
        Returns a trained transform which reverse the target
        after a predictor.
        """
        self._check_is_fitted()
        res = PermutationReciprocalTransformer(
            self.random_state, closest=self.closest)
        res.permutation_ = {v: k for k, v in self.permutation_.items()}
        return res

    def _find_closest(self, cl):
        if not hasattr(self, 'knn_'):
            self.knn_ = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            self.knn_perm_ = numpy.array(list(self.permutation_))
            self.knn_perm_ = self.knn_perm_.reshape((len(self.knn_perm_), 1))
            self.knn_.fit(self.knn_perm_)
        ind = self.knn_.kneighbors([[cl]], return_distance=False)
        res = self.knn_perm_[ind, 0]
        if self.knn_perm_.dtype in (numpy.float32, numpy.float64):
            return float(res)
        if self.knn_perm_.dtype in (numpy.int32, numpy.int64):
            return int(res)
        raise NotImplementedError(  # pragma: no cover
            "The function does not work for type {}.".format(
                self.knn_perm_.dtype))

    def transform(self, X, y):
        """
        Transforms *X* and *y*.
        Returns transformed *X* and *y*.
        If *y* is None, the returned value for *y*
        is None as well.
        """
        if y is None:
            return X, None
        self._check_is_fitted()
        if len(y.shape) == 1 or y.dtype in (numpy.str, numpy.int32, numpy.int64):
            # permutes classes
            yp = y.copy().ravel()
            num = numpy.issubdtype(y.dtype, numpy.floating)
            for i in range(len(yp)):  # pylint: disable=C0200
                if num and numpy.isnan(yp[i]):
                    continue
                if yp[i] not in self.permutation_:
                    if self.closest:
                        cl = self._find_closest(yp[i])
                    else:
                        raise RuntimeError("Unable to find key '{}' in {}.".format(
                            yp[i], list(sorted(self.permutation_))))
                else:
                    cl = yp[i]
                yp[i] = self.permutation_[cl]
            return X, yp.reshape(y.shape)
        else:
            # y is probababilies or raw score
            if len(y.shape) != 2:
                raise RuntimeError(
                    "yp should be a matrix but has shape {}.".format(y.shape))
            cl = [(v, k) for k, v in self.permutation_.items()]
            cl.sort()
            new_perm = {}
            for cl, current in cl:
                new_perm[current] = len(new_perm)
            yp = y.copy()
            for i in range(y.shape[1]):
                yp[:, new_perm[i]] = y[:, i]
            return X, yp
