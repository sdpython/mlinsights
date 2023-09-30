"""
.. _l-search-images-torch-example:

Search images with deep learning (torch)
========================================

Images are usually very different if we compare them at pixel level but
that's quite different if we look at them after they were processed by a
deep learning model. We convert each image into a feature vector
extracted from an intermediate layer of the network.

Get a pre-trained model
-----------------------

We choose the model described in paper `SqueezeNet: AlexNet-level
accuracy with 50x fewer parameters and <0.5MB model
size <https://arxiv.org/abs/1602.07360>`_.
"""

import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset
from mlinsights.ext_test_case import unzip_files
from mlinsights.plotting import plot_gallery_images
from torchvision.models.squeezenet import SqueezeNet1_0_Weights


model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)
model


######################################################################
# The model is stored here:


path = os.path.join(
    os.environ.get("USERPROFILE", os.environ.get("HOME", ".")),
    ".cache",
    "torch",
    "checkpoints",
)
if os.path.exists(path):
    res = os.listdir(path)
else:
    res = ["not found", path]
res


######################################################################
# `pytorch <https://pytorch.org/>`_\ 's design relies on two methods
# *forward* and *backward* which implement the propagation and
# backpropagation of the gradient, the structure is not known and could
# even be dyanmic. That's why it is difficult to define a number of
# layers.


len(model.features), len(model.classifier)


######################################################################
# Images
# ------
#
# We collect images from `pixabay <https://pixabay.com/>`_.
#
# Raw images
# ~~~~~~~~~~


if not os.path.exists("simages/category"):
    os.makedirs("simages/category")

url = "https://github.com/sdpython/mlinsights/raw/ref/_doc/examples/data/dog-cat-pixabay.zip"
files = unzip_files(url, where_to="simages/category")
if len(files) == 0:
    raise FileNotFoundError(f"No images where unzipped from {url!r}.")
len(files), files[0]

##########################################
#

plot_gallery_images(files[:2])

#############################################
#

trans = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # essayer avec 224 seulement
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
imgs = datasets.ImageFolder("simages", trans)
imgs

#######################################
#


dataloader = DataLoader(imgs, batch_size=1, shuffle=False, num_workers=1)
dataloader

#######################################
#
img_seq = iter(dataloader)
img, cl = next(img_seq)

#######################################
#
type(img), type(cl)

#######################################
#
array = img.numpy().transpose((2, 3, 1, 0))
array.shape

#######################################
#

plt.imshow(array[:, :, :, 0])
plt.axis("off")

#######################################
#
img, cl = next(img_seq)
array = img.numpy().transpose((2, 3, 1, 0))
plt.imshow(array[:, :, :, 0])
plt.axis("off")


######################################################################
# `torch <https://pytorch.org/>`_ implements optimized function to load
# and process images.


trans = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # essayer avec 224 seulement
        transforms.RandomRotation((-10, 10), expand=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
imgs = datasets.ImageFolder("simages", trans)
dataloader = DataLoader(imgs, batch_size=1, shuffle=True, num_workers=1)
img_seq = iter(dataloader)
imgs = list(img[0] for i, img in zip(range(2), img_seq))
#######################################
#

plot_gallery_images([img.numpy().transpose((2, 3, 1, 0))[:, :, :, 0] for img in imgs])


######################################################################
# We can multiply the data by implementing a custom
# `sampler <https://github.com/keras-team/keras/issues/7359>`_ or just
# concatenate loaders.


trans1 = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # essayer avec 224 seulement
        transforms.RandomRotation((-10, 10), expand=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
trans2 = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # essayer avec 224 seulement
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
imgs1 = datasets.ImageFolder("simages", trans1)
imgs2 = datasets.ImageFolder("simages", trans2)
dataloader = DataLoader(
    ConcatDataset([imgs1, imgs2]), batch_size=1, shuffle=True, num_workers=1
)
img_seq = iter(dataloader)
imgs = list(img[0] for i, img in zip(range(10), img_seq))
#######################################
#

plot_gallery_images([img.numpy().transpose((2, 3, 1, 0))[:, :, :, 0] for img in imgs])


######################################################################
# Which leaves 52 images to process out of 61 = 31*2 (the folder contains
# 31 images).


len(list(img_seq))


######################################################################
# Search among images
# -------------------
#
# We use the class ``SearchEnginePredictionImages``.


######################################################################
# The idea of the search engine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The deep network is able to classify images coming from a competition
# called `ImageNet <http://image-net.org/>`_ which was trained to
# classify different images. But still, the network has 88 layers which
# slightly transform the images into classification results. We assume the
# last layers contains information which allows the network to classify
# into objects: it is less related to the images than the content of it.
# In particular, we would like that an image with a daark background does
# not necessarily return images with a dark background.

# We reshape an image into *(224x224)* which is the size the network
# ingests. We propagate the inputs until the layer just before the last
# one. Its output will be considered as the *featurized image*. We do that
# for a specific set of images called the *neighbors*. When a new image
# comes up, we apply the same process and find the closest images among
# the set of neighbors.


model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)


######################################################################
# The model outputs the probability for each class.


res = model.forward(imgs[1])
res.shape
#######################################
#

res.detach().numpy().ravel()[:10]
#######################################
#

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot(res.detach().numpy().ravel(), ".")
ax[0].set_title("Output of SqueezeNet")
ax[1].imshow(imgs[1].numpy().transpose((2, 3, 1, 0))[:, :, :, 0])
ax[1].axis("off")


######################################################################
# We have features for one image. We build the neighbors, the output for
# each image in the training datasets.


trans = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.CenterCrop(224), transforms.ToTensor()]
)
imgs = datasets.ImageFolder("simages", trans)
dataloader = DataLoader(imgs, batch_size=1, shuffle=False, num_workers=1)
img_seq = iter(dataloader)
imgs = list(img[0] for img in img_seq)

all_outputs = [model.forward(img).detach().numpy().ravel() for img in imgs]

#######################################
#


knn = NearestNeighbors()
knn.fit(all_outputs)


######################################################################
# We extract the neighbors for a new image.


one_output = model.forward(imgs[5]).detach().numpy().ravel()

score, index = knn.kneighbors([one_output])
score, index


######################################################################
# We need to retrieve images for indexes stored in *index*.


names = os.listdir("simages/category")
names = [os.path.join("simages/category", n) for n in names if ".zip" not in n]
disp = [names[5]] + [names[i] for i in index.ravel()]
disp


######################################################################
# We check the first one is exactly the same as the searched image.


plot_gallery_images(disp)


######################################################################
# It is possible to access intermediate layers output however it means
# rewriting the method forward to capture it: `Accessing intermediate
# layers of a pretrained network
# forward? <https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2>`_.
#
# Going further
# -------------
#
# The original neural network has not been changed and was chosen to be
# small (88 layers). Other options are available for better performances.
# The imported model can be also be trained on a classification problem if
# there is such information to leverage. Even if the model was trained on
# millions of images, a couple of thousands are enough to train the last
# layers. The model can also be trained as long as there exists a way to
# compute a gradient. We could imagine to label the result of this search
# engine and train the model on pairs of images ranked in the other.
#
# We can use the `pairwise
# transform <http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/>`_
# (example of code:
# `ranking.py <https://gist.github.com/fabianp/2020955>`_). For every
# pair :math:`(X_i, X_j)`, we tell if the search engine should have
# :math:`X_i \prec X_j` (:math:`Y_{ij} = 1`) or the order order
# (:math:`Y_{ij} = 0`). :math:`X_i` is the features produced by the neural
# network : :math:`X_i = f(\Omega, img_i)`. We train a classifier on the
# database:
#
# .. math::
#
#       (f(\Omega, img_i) - f(\Omega, img_j), Y_{ij})_{ij}
#
# A training algorithm based on a gradient will have to propagate the gradient:
#
# .. math::
#
#       \frac{\partial f}{\partial \Omega}(img_i) -
#       \frac{\partial f}{\partial \Omega}(img_j)
