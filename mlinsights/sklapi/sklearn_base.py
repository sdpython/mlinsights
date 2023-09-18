# -*- coding: utf-8 -*-
from typing import Any, Dict
import textwrap
import warnings
from .sklearn_parameters import SkLearnParameters


class SkBase:
    """
    Pattern of a *learner* or a *transform* which follows the API
    of :epkg:`scikit-learn`.
    """

    def __init__(self, **kwargs):
        """
        Stores the parameters, see
        :class:`SkLearnParameters`, it keeps a copy of
        the parameters to easily implements method *get_params*
        and clones a model.
        """
        self.P = SkLearnParameters(**kwargs)

    def fit(self, X, y=None, sample_weight=None):
        """
        Trains a model.

        @param      X               features
        @param      y               target
        @param      sample_weight   weight
        @return                     self
        """
        raise NotImplementedError()

    def get_params(self, deep=True):
        """
        Returns the parameters which define the objet,
        all are needed to clone the object.

        @param      deep        unused here
        @return                 dict
        """
        return self.P.to_dict()

    def set_params(self, **values):
        """
        Udpates parameters which define the object,
        all needed to clone the object.

        @param      values      values
        @return                 dictionary
        """
        self.P = SkLearnParameters(**values)
        return self

    def __eq__(self, o):
        """
        Compares two objects, more precisely,
        compares the parameters which define the object.
        """
        return self.test_equality(o, False)

    def test_equality(self, o, exc=True):
        """
        Compares two objects and checks parameters have
        the same values.

        @param      p1      dictionary
        @param      p2      dictionary
        @param      exc     raises an exception if there is a difference
        @return             boolean
        """
        if self.__class__ != o.__class__:
            return False
        p1 = self.get_params()
        p2 = o.get_params()
        return SkBase.compare_params(p1, p2, exc=exc)

    @staticmethod
    def compare_params(
        p1: Dict[str, Any], p2: Dict[str, Any], exc: bool = True
    ) -> bool:
        """
        Compares two sets of parameters.

        :param p1: dictionary
        :param p2: dictionary
        :param exc: raises an exception if error is met
        :return: boolean
        """
        if p1 == p2:
            return True
        for k in p1:
            if k not in p2:
                if exc:
                    raise KeyError(f"Key '{k}' was removed.")
                return False
        for k in p2:
            if k not in p1:
                if exc:
                    raise KeyError(f"Key '{k}' was added.")
                return False
        for k in sorted(p1):
            v1, v2 = p1[k], p2[k]
            if hasattr(v1, "test_equality"):
                b = v1.test_equality(v2, exc=exc)
                if exc and v1 is not v2:
                    warnings.warn(
                        f"v2 is a clone of v1 not v1 itself for key "
                        f"{k!r} and class {type(v1)}."
                    )
            elif isinstance(v1, list) and isinstance(v2, list) and len(v1) == len(v2):
                b = True
                for e1, e2 in zip(v1, v2):
                    if hasattr(e1, "test_equality"):
                        b = e1.test_equality(e2, exc=exc)
                        if not b:
                            return b
            elif isinstance(v1, dict) and isinstance(v2, dict) and set(v1) == set(v2):
                b = True
                for e1, e2 in zip(sorted(v1.items()), sorted(v2.items())):
                    if hasattr(e1[1], "test_equality"):
                        b = e1[1].test_equality(e2[1], exc=exc)
                        if not b:
                            return b
                    elif e1[1] != e2[1]:
                        return False
            elif hasattr(v1, "get_params") and hasattr(v2, "get_params"):
                b = SkBase.compare_params(
                    v1.get_params(deep=False), v2.get_params(deep=False), exc=exc
                )
            else:
                b = v1 == v2
            if not b:
                if exc:
                    raise ValueError(
                        f"Values for key '{k}' are different.\n---\n{v1}\n---\n{v2}"
                    )
                return False
        return True

    def __repr__(self):
        """
        usual
        """
        res = f"{self.__class__.__name__}({str(self.P)})"
        return "\n".join(textwrap.wrap(res, subsequent_indent="    "))
