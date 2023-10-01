"""
.. _l-visualize-pipeline-example:

Visualize a scikit-learn pipeline
=================================

Pipeline can be big with *scikit-learn*, let's dig into a visual way to
look a them.

Simple model
------------

Let's vizualize a simple pipeline, a single model not even trained.
"""

from numpy.random import randn
import pandas
from PIL import Image
from sphinx_runpython.runpython import run_cmd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
)
from mlinsights.helpers.pipeline import (
    alter_pipeline_for_debugging,
    enumerate_pipeline_models,
)
from mlinsights.plotting import pipeline2dot, pipeline2str


iris = datasets.load_iris()
X = iris.data[:, :4]
df = pandas.DataFrame(X)
df.columns = ["X1", "X2", "X3", "X4"]
clf = LogisticRegression()
clf

######################################################################
# The trick consists in converting the pipeline in a graph through the
# `DOT <https://en.wikipedia.org/wiki/DOT_(graph_description_language)>`_
# language.


dot = pipeline2dot(clf, df)
print(dot)


######################################################################
# It is lot better with an image.


dot_file = "graph.dot"
with open(dot_file, "w", encoding="utf-8") as f:
    f.write(dot)


########################################
#


cmd = "dot -G=300 -Tpng {0} -o{0}.png".format(dot_file)
run_cmd(cmd, wait=True)


img = Image.open("graph.dot.png")
img


######################################################################
# Complex pipeline
# ----------------
#
# *scikit-learn* instroduced a couple of transform to play with features
# in a single pipeline. The following example is taken from `Column
# Transformer with Mixed
# Types <https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py>`_.


columns = [
    "pclass",
    "name",
    "sex",
    "age",
    "sibsp",
    "parch",
    "ticket",
    "fare",
    "cabin",
    "embarked",
    "boat",
    "body",
    "home.dest",
]

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="lbfgs")),
    ]
)
clf


######################################################################
# Let's see it first as a simplified text.


print(pipeline2str(clf))

########################################
#


dot = pipeline2dot(clf, columns)

dot_file = "graph2.dot"
with open(dot_file, "w", encoding="utf-8") as f:
    f.write(dot)

cmd = "dot -G=300 -Tpng {0} -o{0}.png".format(dot_file)
run_cmd(cmd, wait=True)

img = Image.open("graph2.dot.png")
img


######################################################################
# Example with FeatureUnion
# -------------------------


model = Pipeline(
    [
        ("poly", PolynomialFeatures()),
        (
            "union",
            FeatureUnion([("scaler2", MinMaxScaler()), ("scaler3", StandardScaler())]),
        ),
    ]
)
dot = pipeline2dot(model, columns)

dot_file = "graph3.dot"
with open(dot_file, "w", encoding="utf-8") as f:
    f.write(dot)

cmd = "dot -G=300 -Tpng {0} -o{0}.png".format(dot_file)
run_cmd(cmd, wait=True)

img = Image.open("graph3.dot.png")
img


######################################################################
# Compute intermediate outputs
# ----------------------------

# It is difficult to access intermediate outputs with *scikit-learn* but
# it may be interesting to do so. The method
# `alter_pipeline_for_debugging <find://alter_pipeline_for_debugging>`_
# modifies the pipeline to intercept intermediate outputs.


model = Pipeline(
    [
        ("scaler1", StandardScaler()),
        (
            "union",
            FeatureUnion([("scaler2", StandardScaler()), ("scaler3", MinMaxScaler())]),
        ),
        ("lr", LinearRegression()),
    ]
)

X = randn(4, 5)
y = randn(4)
model.fit(X, y)
########################################
#

print(pipeline2str(model))


######################################################################
# Let's now modify the pipeline to get the intermediate outputs.


alter_pipeline_for_debugging(model)


######################################################################
# The function adds a member ``_debug`` which stores inputs and outputs in
# every piece of the pipeline.


model.steps[0][1]._debug
########################################
#

model.predict(X)


######################################################################
# The member was populated with inputs and outputs.


model.steps[0][1]._debug


######################################################################
# Every piece behaves the same way.


for coor, model, vars in enumerate_pipeline_models(model):
    print(coor)
    print(model._debug)
