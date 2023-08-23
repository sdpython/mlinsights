# -*- coding: utf-8 -*-
"""
@file
@brief Module *mlinsights*.
Look for insights for machine learned models.
"""
__version__ = "0.5.0"
__author__ = "Xavier Dupré"
__github__ = "https://github.com/sdpython/mlinsights"
__url__ = "https://sdpython.github.io/doc/dev/mlinsights/"
__license__ = "MIT License"



def check(log=False):
    """
    Checks the library is working.
    It raises an exception.
    If you want to disable the logs:

    @param      log     if True, display information, otherwise
    @return             0 or exception
    """
    return True  # pragma: no cover


def _setup_hook(use_print=False):
    """
    if this function is added to the module,
    the help automation and unit tests call it first before
    anything goes on as an initialization step.
    """
    # we can check many things, needed module
    # any others things before unit tests are started
    if use_print:  # pragma: no cover
        print("Success: _setup_hook")  # pragma: no cover
