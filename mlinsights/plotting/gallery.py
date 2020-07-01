"""
@file
@brief Featurizers for machine learned models.
"""
import sys
import io
import os
import urllib.request
from PIL import Image


def plot_gallery_images(imgs, texts=None, width=4, return_figure=False,
                        ax=None, folder_image=None, **figure):
    """
    Plots a gallery of images using :epkg:`matplotlib`.

    @param      imgs            list of images (filename, urls or :epkg:`Pillow` objects),
    @param      texts           text to display (if None, print ``'img % i'``)
    @param      width           number of images on the same line (unused if *imgs* is a matrix)
    @param      figure          additional parameters when the figure is created
    @param      return_figure   return the figure as well as the axes
    @param      ax              None or existing axes, it should have the same
                                shape of *imgs*
    @param      folder_image    image paths may be relative to some folder,
                                in that case, they should be relative to
                                this folder
    @return                     axes or (figure, axes) if *return_figure* is True

    .. image:: gal.jpg

    See also notebook :ref:`searchimageskerasrst` or
    :ref:`searchimagestorchrst` for an example.
    *imgs* can also be a matrix to force the function to
    use the same coordinates.
    """
    if "plt" not in sys.modules:
        import matplotlib.pyplot as plt  # pylint: disable=C0415

    if hasattr(imgs, 'shape') and len(imgs.shape) == 2:
        height, width = imgs.shape
        if ax is not None and ax.shape != imgs.shape:
            raise ValueError(  # pragma: no cover
                "ax.shape {0} != imgs.shape {1}".format(ax.shape, imgs.shape))
        imgs = imgs.ravel()
        if texts is not None:
            texts = texts.ravel()
    else:
        height = len(imgs) // width
        if len(imgs) % width != 0:
            height += 1

    if ax is None:
        if 'figsize' not in figure:
            figure['figsize'] = (12, height * 3)
        fig, ax = plt.subplots(height, width, **figure)
    elif return_figure:
        raise ValueError(  # pragma: no cover
            "ax is specified and return_figure is True")

    for i, img in enumerate(imgs):
        if img is None:
            continue
        y, x = i // width, i % width
        if height == 1:
            ind = x
        elif width == 1:
            ind = y
        else:
            ind = y, x

        if isinstance(img, str):
            if "//" in img:
                # url
                with urllib.request.urlopen(img) as response:
                    content = response.read()
                try:
                    im = Image.open(io.BytesIO(content))
                except OSError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to read image '{}'.".format(img)) from e
            else:
                # local file
                if folder_image is not None:
                    im = Image.open(os.path.join(folder_image, img))
                else:
                    im = Image.open(img)
        else:
            im = img
        if hasattr(im, 'size'):
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
    return ax
