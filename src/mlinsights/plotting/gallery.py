"""
@file
@brief Featurizers for machine learned models.
"""
import sys
import io
import urllib.request
from PIL import Image


def plot_gallery_images(imgs, texts=None, width=4, return_figure=False, **figure):
    """
    Plots a gallery of images using :epkg:`matplotlib`.

    @param      imgs            list of images (filename, urls or :epkg:`Pillow` objects),
    @param      texts           text to display (if None, print ``'img % i'``)
    @param      width           number of images on the same line
    @param      figure          additional parameters when the figure is created
    @param      return_figure   return the figure as well as the axes
    @return                     axes or (figure, axes) if *return_figure* is True

    See notebook :ref:`searchimagesrst` for an example.
    """
    if "plt" not in sys.modules:
        import matplotlib.pyplot as plt

    height = len(imgs) // width
    if len(imgs) % width != 0:
        height += 1

    if 'figsize' not in figure:
        figure['figsize'] = (12, height * 3)

    fig, ax = plt.subplots(height, width, **figure)

    for i, img in enumerate(imgs):
        y, x = i // width, i % width
        if height == 1:
            ind = x
        else:
            ind = y, x

        if isinstance(img, str):
            if "//" in img:
                # url
                with urllib.request.urlopen(img) as response:
                    content = response.read()
                im = Image.open(io.BytesIO(content))
            else:
                # local file
                im = Image.open(img)
        else:
            im = img
        ax[ind].imshow(im)
        if texts is None:
            t = "img %d" % i
        else:
            t = texts[i]
        ax[ind].text(0, 0, t)
        ax[ind].axis('off')

    for i in range(len(imgs), width * height):
        y, x = i // width, i % width
        if height == 1:
            ind = x
        else:
            ind = y, x
        ax[ind].axis('off')

    if return_figure:
        return fig, ax
    else:
        return ax
