# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import os
import unittest
import warnings
import http.client
import numpy
from pyquickhelper.loghelper import noLOG
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.filehelper import unzip_files
from pyquickhelper.pycode import fix_tkinter_issues_virtualenv
from mlinsights.plotting import plot_gallery_images


class TestPlotGallery(ExtTestCase):

    def test_plot_gallery(self):
        temp = get_temp_folder(__file__, "temp_plot_gallery")
        zipimg = os.path.join(temp, "..", "..", "..", "_doc",
                              "notebooks", "explore", "data", "dog-cat-pixabay.zip")
        files = unzip_files(zipimg, where_to=temp)

        fix_tkinter_issues_virtualenv(fLOG=noLOG)
        from matplotlib import pyplot as plt

        fig, _ = plot_gallery_images(files[:2], return_figure=True)
        img = os.path.join(temp, "gallery.png")
        fig.savefig(img)
        plt.close('all')

    def test_plot_gallery_matrix(self):
        temp = get_temp_folder(__file__, "temp_plot_gallery_matrix")
        zipimg = os.path.join(temp, "..", "..", "..", "_doc",
                              "notebooks", "explore", "data", "dog-cat-pixabay.zip")
        files = unzip_files(zipimg, where_to=temp)

        fix_tkinter_issues_virtualenv(fLOG=noLOG)
        from matplotlib import pyplot as plt

        fig, _ = plot_gallery_images(numpy.array(
            files[:2]).reshape((2, 1)), return_figure=True)
        img = os.path.join(temp, "gallery.png")
        fig.savefig(img)
        plt.close('all')

    def test_plot_gallery_url(self):
        fix_tkinter_issues_virtualenv(fLOG=noLOG)
        from matplotlib import pyplot as plt

        root = "http://www.xavierdupre.fr/enseignement/complements/dog-cat-pixabay/"
        files = [root + 'cat-2603300__480.jpg',
                 root + 'cat-2947188__480.jpg']

        temp = get_temp_folder(__file__, "temp_plot_gallery_url")
        try:
            fig, ax = plot_gallery_images(files, return_figure=True)
        except http.client.RemoteDisconnected as e:
            warnings.warn(f"Unable to fetch image {e}'")
            return
        img = os.path.join(temp, "gallery.png")
        fig.savefig(img)
        plt.close('all')

        # ax
        try:
            ax = plot_gallery_images(files, return_figure=False, ax=ax)
            self.assertNotEmpty(ax)
        except http.client.RemoteDisconnected as e:
            warnings.warn(f"Unable to fetch image {e}'")
            return


if __name__ == "__main__":
    unittest.main()
