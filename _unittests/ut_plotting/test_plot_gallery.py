# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""

import sys
import os
import unittest


try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.filehelper import unzip_files
from pyquickhelper.pycode import fix_tkinter_issues_virtualenv
from src.mlinsights.plotting import plot_gallery_images


class TestPlotGallery(ExtTestCase):

    def test_plot_gallery(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_plot_gallery")
        zipimg = os.path.join(temp, "..", "..", "..", "_doc",
                              "notebooks", "data", "dog-cat-pixabay.zip")
        files = unzip_files(zipimg, where_to=temp)

        fix_tkinter_issues_virtualenv(fLOG=fLOG)
        from matplotlib import pyplot as plt

        fig, ax = plot_gallery_images(files[:2], return_figure=True)
        img = os.path.join(temp, "gallery.png")
        fig.savefig(img)
        plt.close('all')

    def test_plot_gallery_url(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        fix_tkinter_issues_virtualenv(fLOG=fLOG)
        from matplotlib import pyplot as plt

        root = "http://www.xavierdupre.fr/enseignement/complements/dog-cat-pixabay/"
        files = [root + 'cat-2603300__480.jpg',
                 root + 'cat-2947188__480.jpg']

        temp = get_temp_folder(__file__, "temp_plot_gallery_url")
        fig, ax = plot_gallery_images(files, return_figure=True)
        img = os.path.join(temp, "gallery.png")
        fig.savefig(img)
        plt.close('all')


if __name__ == "__main__":
    unittest.main()
