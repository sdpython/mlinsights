# -*- coding: utf-8 -*-
import textwrap


class SkException(Exception):

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

        :param name: name
        :param value: value
        :raise: raises :class:`SkException` if error
        """
        if name.startswith("_") or name.endswith("_"):
            raise SkException(f"Parameter name must not start by '_': '{name}'")

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
                return f"'{v}'"
            return repr(v)

        text = ", ".join(f"{k}={fmt(getattr(self, k))}" for k in sorted(self.Keys))
        return "\n".join(textwrap.wrap(text, subsequent_indent="    "))

    def to_dict(self):
        """
        Returns parameters as a dictionary.

        @return         dict
        """
        return {k: getattr(self, k) for k in self.Keys}
