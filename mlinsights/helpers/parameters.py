import textwrap


def format_value(v):
    """
    Formats a value to be included in a string.

    :param v: a string
    :return: a string
    """
    return "%r" % v.replace("'", "\\'") if isinstance(v, str) else f"{v!r}"


def format_parameters(pdict):
    """
    Formats a list of parameters.

    :param pdict: dictionary
    :return: string

    .. runpython::
        :showcode:

        from mlinsights.helpers.parameters import format_parameters

        d = dict(i=2, x=6.7, s="r")
        print(format_parameters(d))
    """
    res = []
    for k, v in sorted(pdict.items()):
        res.append(f"{k}={format_value(v)}")
    return ", ".join(res)


def format_function_call(name, pdict):
    """
    Formats a function call with named parameters.

    param pdict: dictionary
    :return: string

    .. runpython::
        :showcode:

        from mlinsights.helpers.parameters import format_function_call

        d = dict(i=2, x=6.7, s="r")
        print(format_function_call("fct", d))
    """
    res = f"{name}({format_parameters(pdict)})"
    return "\n".join(textwrap.wrap(res, width=70, subsequent_indent="    "))
