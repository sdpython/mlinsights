"""
.. _l-logisitic-regression-clustering:

LogisticRegression and Clustering
=================================

A logistic regression implements a convex partition of the features
spaces. A clustering algorithm applied before the trainer modifies the
feature space in way the partition is not necessarily convex in the
initial features. Let's see how.

A dummy datasets and not convex
-------------------------------
"""

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlinsights.mlmodel import ClassifierAfterKMeans

Xs = []
Ys = []
n = 20
for i in range(5):
    for j in range(4):
        x1 = numpy.random.rand(n) + i * 1.1
        x2 = numpy.random.rand(n) + j * 1.1
        Xs.append(numpy.vstack([x1, x2]).T)
        cl = numpy.random.randint(0, 4)
        Ys.extend([cl for i in range(n)])
X = numpy.vstack(Xs)
Y = numpy.array(Ys)
X.shape, Y.shape, set(Y)

########################################
#


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i in set(Y):
    ax.plot(
        X[i == Y, 0], X[i == Y, 1], "o", label="cl%d" % i, color=plt.cm.tab20.colors[i]
    )
ax.legend()
ax.set_title("Classification not convex")


######################################################################
# One function to plot classification in 2D
# -----------------------------------------


def draw_border(
    clr,
    X,
    y,
    fct=None,
    incx=1,
    incy=1,
    figsize=None,
    border=True,
    clusters=None,
    ax=None,
):
    # see https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    # https://matplotlib.org/examples/color/colormaps_reference.html

    h = 0.02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - incx, X[:, 0].max() + incx
    y_min, y_max = X[:, 1].min() - incy, X[:, 1].max() + incy
    xx, yy = numpy.meshgrid(
        numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h)
    )
    if fct is None:
        Z = clr.predict(numpy.c_[xx.ravel(), yy.ravel()])
    else:
        Z = fct(clr, numpy.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    cmap = plt.cm.tab20
    Z = Z.reshape(xx.shape)
    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=figsize or (4, 3))
    ax.pcolormesh(xx, yy, Z, cmap=cmap)

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=cmap)
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    # Plot clusters
    if clusters is not None:
        mat = []
        ym = []
        for k, v in clusters.items():
            mat.append(v.cluster_centers_)
            ym.extend(k for i in range(v.cluster_centers_.shape[0]))
        cx = numpy.vstack(mat)
        ym = numpy.array(ym)
        ax.scatter(cx[:, 0], cx[:, 1], c=ym, edgecolors="y", cmap=cmap, s=300)
    return ax


######################################################################
# Logistic Regression
# -------------------


clr = LogisticRegression(solver="lbfgs")
clr.fit(X, Y)

########################################
#


ax = draw_border(clr, X, Y, incx=1, incy=1, figsize=(6, 4), border=False)
ax.set_title("Logistic Regression")


######################################################################
# Not quite close!


######################################################################
# Logistic Regression and k-means
# -------------------------------


clk = ClassifierAfterKMeans(e_solver="lbfgs")
clk.fit(X, Y)


######################################################################
# The centers of the first k-means:


clk.clus_[0].cluster_centers_

########################################
#


ax = draw_border(
    clk, X, Y, incx=1, incy=1, figsize=(6, 4), border=False, clusters=clk.clus_
)
ax.set_title("Logistic Regression and K-Means - 2 clusters per class")


######################################################################
# The big cricles are the centers of the k-means fitted for each class. It
# look better!


######################################################################
# Variation
# ---------


dt = []
for cl in range(1, 6):
    clk = ClassifierAfterKMeans(c_n_clusters=cl, e_solver="lbfgs", e_max_iter=700)
    clk.fit(X, Y)
    sc = clk.score(X, Y)
    dt.append(dict(score=sc, nb_clusters=cl))


pandas.DataFrame(dt)

########################################
#


ax = draw_border(
    clk, X, Y, incx=1, incy=1, figsize=(6, 4), border=False, clusters=clk.clus_
)
ax.set_title("Logistic Regression and K-Means - 8 clusters per class")


######################################################################
# Random Forest
# -------------

# The random forest works without any clustering as expected.


rf = RandomForestClassifier(n_estimators=20)
rf.fit(X, Y)

########################################
#


ax = draw_border(rf, X, Y, incx=1, incy=1, figsize=(6, 4), border=False)
ax.set_title("Random Forest")
