"""
@file
@brief run all unit tests
"""

import os
import sys


def main():
    try:
        import pyquickhelper as skip_
    except ImportError:
        sys.path.append(
            os.path.normpath(
                os.path.abspath(
                    os.path.join(
                        os.path.split(__file__)[0],
                        "..",
                        "..",
                        "pyquickhelper",
                        "src"))))
        if "PYQUICKHELPER" in os.environ and len(os.environ["PYQUICKHELPER"]) > 0:
            sys.path.append(os.environ["PYQUICKHELPER"])
        import pyquickhelper as skip_

    from pyquickhelper.loghelper import fLOG
    from pyquickhelper.pycode import main_wrapper_tests
    fLOG(OutputPrint=True)
    main_wrapper_tests(__file__)


if __name__ == "__main__":
    main()
