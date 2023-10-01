"""
Piecewise linear regression with scikit-learn predictors
========================================================

The notebook illustrates an implementation of a piecewise linear
regression based on
`scikit-learn <https://scikit-learn.org/stable/index.html>`_. The
bucketization can be done with a
`DecisionTreeRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_
or a
`KBinsDiscretizer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html>`_.
A linear model is then fitted on each bucket.

Piecewise data
--------------

Let's build a toy problem based on two linear models.
"""

import numpy
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.dummy import DummyRegressor
from mlinsights.mlmodel import PiecewiseRegressor


X = npr.normal(size=(1000, 4))
alpha = [4, -2]
t = (X[:, 0] + X[:, 3] * 0.5) > 0
switch = numpy.zeros(X.shape[0])
switch[t] = 1
y = alpha[0] * X[:, 0] * t + alpha[1] * X[:, 0] * (1 - t) + X[:, 2]

########################################
#


fig, ax = plt.subplots(1, 1)
ax.plot(X[:, 0], y, ".")
ax.set_title("Piecewise examples")


######################################################################
# Piecewise Linear Regression with a decision tree
# ------------------------------------------------
#
# The first example is done with a decision tree.


X_train, X_test, y_train, y_test = train_test_split(X[:, :1], y)

########################################
#


model = PiecewiseRegressor(
    verbose=True, binner=DecisionTreeRegressor(min_samples_leaf=300)
)
model.fit(X_train, y_train)

########################################
#


pred = model.predict(X_test)
pred[:5]

########################################
#


fig, ax = plt.subplots(1, 1)
ax.plot(X_test[:, 0], y_test, ".", label="data")
ax.plot(X_test[:, 0], pred, ".", label="predictions")
ax.set_title("Piecewise Linear Regression\n2 buckets")
ax.legend()


######################################################################
# The method *transform_bins* returns the bucket of each variables, the
# final leave from the tree.


model.transform_bins(X_test)


######################################################################
# Let's try with more buckets.


model = PiecewiseRegressor(
    verbose=False, binner=DecisionTreeRegressor(min_samples_leaf=150)
)
model.fit(X_train, y_train)

########################################
#


fig, ax = plt.subplots(1, 1)
ax.plot(X_test[:, 0], y_test, ".", label="data")
ax.plot(X_test[:, 0], model.predict(X_test), ".", label="predictions")
ax.set_title("Piecewise Linear Regression\n4 buckets")
ax.legend()


######################################################################
# Piecewise Linear Regression with a KBinsDiscretizer
# ---------------------------------------------------


model = PiecewiseRegressor(verbose=True, binner=KBinsDiscretizer(n_bins=2))
model.fit(X_train, y_train)

########################################
#


fig, ax = plt.subplots(1, 1)
ax.plot(X_test[:, 0], y_test, ".", label="data")
ax.plot(X_test[:, 0], model.predict(X_test), ".", label="predictions")
ax.set_title("Piecewise Linear Regression\n2 buckets")
ax.legend()

########################################
#


model = PiecewiseRegressor(verbose=True, binner=KBinsDiscretizer(n_bins=4))
model.fit(X_train, y_train)

########################################
#


fig, ax = plt.subplots(1, 1)
ax.plot(X_test[:, 0], y_test, ".", label="data")
ax.plot(X_test[:, 0], model.predict(X_test), ".", label="predictions")
ax.set_title("Piecewise Linear Regression\n4 buckets")
ax.legend()


######################################################################
# The model does not enforce continuity despite the fast it looks like so.
# Let's compare with a constant on each bucket.


model = PiecewiseRegressor(
    verbose="tqdm", binner=KBinsDiscretizer(n_bins=4), estimator=DummyRegressor()
)
model.fit(X_train, y_train)

########################################
#


fig, ax = plt.subplots(1, 1)
ax.plot(X_test[:, 0], y_test, ".", label="data")
ax.plot(X_test[:, 0], model.predict(X_test), ".", label="predictions")
ax.set_title("Piecewise Constants\n4 buckets")
ax.legend()


######################################################################
# Next
# ----

# PR `Model trees (M5P and
# co) <https://github.com/scikit-learn/scikit-learn/issues/13106>`_ and
# issue `Model trees
# (M5P) <https://github.com/scikit-learn/scikit-learn/pull/13732>`_
# propose an implementation a piecewise regression with any kind of
# regression model. It is based on `Building Model
# Trees <https://github.com/ankonzoid/LearningX/tree/master/advanced_ML/model_tree%3E>`_.
# It fits many models to find the best splits.
