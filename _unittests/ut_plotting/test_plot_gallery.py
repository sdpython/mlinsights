import os
import tempfile
import unittest
import warnings
import http.client
import urllib.error
import numpy
from mlinsights.ext_test_case import ExtTestCase, unzip_files
from mlinsights.plotting import plot_gallery_images


class TestPlotGallery(ExtTestCase):
    def test_plot_gallery(self):
        this = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as temp:
            zipimg = os.path.join(
                this,
                "..",
                "..",
                "_doc",
                "examples",
                "data",
                "dog-cat-pixabay.zip",
            )
            files = unzip_files(zipimg, where_to=temp)

            from matplotlib import pyplot as plt

            fig, _ = plot_gallery_images(files[:2], return_figure=True)
            img = os.path.join(temp, "gallery.png")
            fig.savefig(img)
            plt.close("all")

    def test_plot_gallery_matrix(self):
        this = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as temp:
            zipimg = os.path.join(
                this,
                "..",
                "..",
                "_doc",
                "examples",
                "data",
                "dog-cat-pixabay.zip",
            )
            files = unzip_files(zipimg, where_to=temp)

            from matplotlib import pyplot as plt

            fig, _ = plot_gallery_images(
                numpy.array(files[:2]).reshape((2, 1)), return_figure=True
            )
            img = os.path.join(temp, "gallery.png")
            fig.savefig(img)
            plt.close("all")

    def test_plot_gallery_url(self):
        from matplotlib import pyplot as plt

        root = "http://www.xavierdupre.fr/enseignement/complements/dog-cat-pixabay/"
        files = [root + "cat-2603300__480.jpg", root + "cat-2947188__480.jpg"]

        try:
            fig, ax = plot_gallery_images(files, return_figure=True)
        except (http.client.RemoteDisconnected, urllib.error.HTTPError) as e:
            raise unittest.SkipTest(f"Unable to fetch image {e} (url={root!r})")
        self.assertNotEmpty(fig)
        self.assertNotEmpty(ax)

        # ax
        try:
            ax = plot_gallery_images(files, return_figure=False, ax=ax)
            self.assertNotEmpty(ax)
        except http.client.RemoteDisconnected as e:
            warnings.warn(f"Unable to fetch image {e}'", stacklevel=0)
            plt.close("all")
            return
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
