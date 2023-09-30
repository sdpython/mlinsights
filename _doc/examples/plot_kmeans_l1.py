"""
.. _l-kmeans-l1-example:

KMeans with norm L1
===================

This demonstrates how results change when using norm L1 for a k-means
algorithm.
"""


import matplotlib.pyplot as plt
import numpy
import numpy.random as rnd
from sklearn.cluster import KMeans
from mlinsights.mlmodel import KMeansL1L2


######################################################################
# Simple datasets
# ---------------


N = 1000
X = numpy.zeros((N * 2, 2), dtype=numpy.float64)
X[:N] = rnd.rand(N, 2)
X[N:] = rnd.rand(N, 2)
# X[N:, 0] += 0.75
X[N:, 1] += 1
X[: N // 10, 0] -= 2
X.shape

########################################
#

fig, ax = plt.subplots(1, 1)
ax.plot(X[:, 0], X[:, 1], ".")
ax.set_title("Two squares")


######################################################################
# Classic KMeans
# --------------
#
# It uses euclidean distance.


km = KMeans(2)
km.fit(X)

km.cluster_centers_


def plot_clusters(km_, X, ax):
    lab = km_.predict(X)
    for i in range(km_.cluster_centers_.shape[0]):
        sub = X[lab == i]
        ax.plot(sub[:, 0], sub[:, 1], ".", label="c=%d" % i)
    C = km_.cluster_centers_
    ax.plot(C[:, 0], C[:, 1], "o", ms=15, label="centers")
    ax.legend()


fig, ax = plt.subplots(1, 1)
plot_clusters(km, X, ax)
ax.set_title("L2 KMeans")


######################################################################
# KMeans with L1 norm
# -------------------


kml1 = KMeansL1L2(2, norm="L1")
kml1.fit(X)

########################################
#


kml1.cluster_centers_

########################################
#

fig, ax = plt.subplots(1, 1)
plot_clusters(kml1, X, ax)
ax.set_title("L1 KMeans")


######################################################################
# When clusters are completely different
# --------------------------------------


N = 1000
X = numpy.zeros((N * 2, 2), dtype=numpy.float64)
X[:N] = rnd.rand(N, 2)
X[N:] = rnd.rand(N, 2)
# X[N:, 0] += 0.75
X[N:, 1] += 1
X[: N // 10, 0] -= 4
X.shape

########################################
#


km = KMeans(2)
km.fit(X)

########################################
#

kml1 = KMeansL1L2(2, norm="L1")
kml1.fit(X)

########################################
#

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_clusters(km, X, ax[0])
plot_clusters(kml1, X, ax[1])
ax[0].set_title("L2 KMeans")
ax[1].set_title("L1 KMeans")
