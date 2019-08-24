"""
@file
@brief Implements a transform which modifies the target
and applies the reverse transformation on the target.
"""
import numpy
from .sklearn_transform_inv import BaseReciprocalTransformer


class FunctionReciprocalTransformer(BaseReciprocalTransformer):
    """
    The transform is used to apply a function on a the target,
    predict, then transform the target back before scoring.
    The transforms implements a series of predefined functions:

    .. runpython::
        :showcode:

        import pprint
        from mlinsight.mlmodel.sklearn_transform_inv_fct import FunctionReciprocalTransformer
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
            'exp(x)-1': (lambda x: numpy.exp(x) - 1, 'log'),
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
                raise ValueError(
                    "If fct is a function name, fct_inv must not be specified.")
            opts = self.__class__.available_fcts()
            if fct not in opts:
                raise ValueError("Unknown fct '{}', it should in {}.".format(
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
        """
        return X, self.fct_(y)
