# -*- coding: utf-8 -*-
"""
@file
@brief Defines class @see cl SkLearnParameters.
"""
import textwrap


class SkException (Exception):

    """
    custom exception
    """
    pass


class SkLearnParameters:

    """
    Defines a class to store parameters of a *learner* or a *transform*.
    """

    def __init__(self, **kwargs):
        """
        Stores parameters as members of the class itself.
        """
        self._keys = list(kwargs.keys())
        for k, v in kwargs.items():
            self.validate(k, v)
            setattr(self, k, v)

    def validate(self, name, value):
        """
        Verifies a parameter and its value.

        @param      name        name
        @param      value       value
        @raises                 raises @see cl SkException if error
        """
        if name.startswith("_") or name.endswith("_"):
            raise SkException(  # pragma: no cover
                "Parameter name must not start by '_': '{0}'".format(name))

    @property
    def Keys(self):
        """
        Returns parameter names.
        """
        return self._keys

    def __repr__(self):
        """
        usual
        """
        def fmt(v):
            "formatting function"
            if isinstance(v, str):
                return "'{0}'".format(v)
            return repr(v)
        text = ", ".join("{0}={1}".format(k, fmt(getattr(self, k)))
                         for k in sorted(self.Keys))
        return "\n".join(textwrap.wrap(text, subsequent_indent="    "))

    def to_dict(self):
        """
        Returns parameters as a dictionary.

        @return         dict
        """
        return {k: getattr(self, k) for k in self.Keys}
