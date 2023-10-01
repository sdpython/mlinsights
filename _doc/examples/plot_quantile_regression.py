"""
.. _l-quantile-regression-example:

Quantile Regression
===================

`scikit-learn <http://scikit-learn.org/stable/>`_ does not have a
quantile regression.
`mlinsights <https://sdpython.github.io/doc/dev/mlinsights/index.html>`_
implements a version of it.

Simple example
--------------

We first generate some dummy data.
"""

import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from mlinsights.mlmodel import QuantileLinearRegression

X = numpy.random.random(1000)
eps1 = (numpy.random.random(900) - 0.5) * 0.1
eps2 = (numpy.random.random(100)) * 10
eps = numpy.hstack([eps1, eps2])
X = X.reshape((1000, 1))
Y = X.ravel() * 3.4 + 5.6 + eps

########################################
#


clr = LinearRegression()
clr.fit(X, Y)

########################################
#


clq = QuantileLinearRegression()
clq.fit(X, Y)


data = dict(X=X.ravel(), Y=Y, clr=clr.predict(X), clq=clq.predict(X))
df = DataFrame(data)
df.head()

########################################
#


fig, ax = plt.subplots(1, 1, figsize=(10, 4))
choice = numpy.random.choice(X.shape[0] - 1, size=100)
xx = X.ravel()[choice]
yy = Y[choice]
ax.plot(xx, yy, ".", label="data")
xx = numpy.array([[0], [1]])
y1 = clr.predict(xx)
y2 = clq.predict(xx)
ax.plot(xx, y1, "--", label="L2")
ax.plot(xx, y2, "--", label="L1")
ax.set_title("Quantile (L1) vs Square (L2)")
ax.legend()


######################################################################
# The L1 is clearly less sensible to extremas. The optimization algorithm
# is based on `Iteratively reweighted least
# squares <https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares>`_.
# It estimates a linear regression with error L2 then reweights each
# oberservation with the inverse of the error L1.


clq = QuantileLinearRegression(verbose=True, max_iter=20)
clq.fit(X, Y)
########################################
#


clq.score(X, Y)


######################################################################
# Regression with various quantiles
# ---------------------------------


X = numpy.random.random(1200)
eps1 = (numpy.random.random(900) - 0.5) * 0.5
eps2 = (numpy.random.random(300)) * 2
eps = numpy.hstack([eps1, eps2])
X = X.reshape((1200, 1))
Y = X.ravel() * 3.4 + 5.6 + eps + X.ravel() * X.ravel() * 8
########################################
#


fig, ax = plt.subplots(1, 1, figsize=(10, 4))
choice = numpy.random.choice(X.shape[0] - 1, size=100)
xx = X.ravel()[choice]
yy = Y[choice]
ax.plot(xx, yy, ".", label="data")
ax.set_title("Almost linear dataset")
########################################
#


clqs = {}
for qu in [0.1, 0.25, 0.5, 0.75, 0.9]:
    clq = QuantileLinearRegression(quantile=qu)
    clq.fit(X, Y)
    clqs["q=%1.2f" % qu] = clq
########################################
#


fig, ax = plt.subplots(1, 1, figsize=(10, 4))
choice = numpy.random.choice(X.shape[0] - 1, size=100)
xx = X.ravel()[choice]
yy = Y[choice]
ax.plot(xx, yy, ".", label="data")
xx = numpy.array([[0], [1]])
for qu in sorted(clqs):
    y = clqs[qu].predict(xx)
    ax.plot(xx, y, "--", label=qu)
ax.set_title("Various quantiles")
ax.legend()
