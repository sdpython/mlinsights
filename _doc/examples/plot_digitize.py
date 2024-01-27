"""
========================
numpy.digitize as a tree
========================

Function :func:`numpy.digitize` transforms a real variable
into a discrete one by returning the buckets the variable
falls into. This bucket can be efficiently retrieved by doing a
binary search over the bins. That's equivalent to decision tree.
Function :func:`digitize2tree
<mlinsights.mltree.tree_digitize.digitize2tree>`.

Simple example
==============
"""

import numpy
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from pandas import DataFrame, pivot, pivot_table
from skl2onnx import to_onnx
from sklearn.tree import export_text
from tqdm import tqdm
from mlinsights.ext_test_case import measure_time
from mlinsights.mltree import digitize2tree

x = numpy.array([0.2, 6.4, 3.0, 1.6])
bins = numpy.array([0.0, 1.0, 2.5, 4.0, 7.0])
expected = numpy.digitize(x, bins, right=True)
tree = digitize2tree(bins, right=True)
pred = tree.predict(x.reshape((-1, 1)))
print(expected, pred)

##########################################
# The tree looks like the following.

print(export_text(tree, feature_names=["x"]))

#######################################
# Benchmark
# =========
#
# Let's measure the processing time. *numpy* should be
# much faster than *scikit-learn* as it is adding many verification.
# However, the benchmark also includes a conversion of the tree into
# ONNX and measure the processing time with :epkg:`onnxruntime`.

obs = []

for shape in tqdm([1, 10, 100, 1000, 10000, 100000]):
    x = numpy.random.random(shape).astype(numpy.float32)
    if shape < 1000:
        repeat = number = 100
    else:
        repeat = number = 10

    for n_bins in [1, 10, 100]:
        bins = (numpy.arange(n_bins) / n_bins).astype(numpy.float32)

        ti = measure_time(
            "numpy.digitize(x, bins, right=True)",
            context={"numpy": numpy, "x": x, "bins": bins},
            div_by_number=True,
            repeat=repeat,
            number=number,
        )
        ti["name"] = "numpy"
        ti["n_bins"] = n_bins
        ti["shape"] = shape
        obs.append(ti)

        tree = digitize2tree(bins, right=True)

        ti = measure_time(
            "tree.predict(x)",
            context={"numpy": numpy, "x": x.reshape((-1, 1)), "tree": tree},
            div_by_number=True,
            repeat=repeat,
            number=number,
        )
        ti["name"] = "sklearn"
        ti["n_bins"] = n_bins
        ti["shape"] = shape
        obs.append(ti)

        onx = to_onnx(tree, x.reshape((-1, 1)), target_opset=15)

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        ti = measure_time(
            "sess.run(None, {'X': x})",
            context={"numpy": numpy, "x": x.reshape((-1, 1)), "sess": sess},
            div_by_number=True,
            repeat=repeat,
            number=number,
        )
        ti["name"] = "ort"
        ti["n_bins"] = n_bins
        ti["shape"] = shape
        obs.append(ti)


df = DataFrame(obs)
piv = pivot_table(
    data=df, index="shape", columns=["n_bins", "name"], values=["average"]
)
print(piv)

##########################################
# Plotting
# ========

n_bins = list(sorted(set(df.n_bins)))
fig, ax = plt.subplots(1, len(n_bins), figsize=(14, 4))

for i, nb in enumerate(n_bins):
    piv = pivot(
        data=df[df.n_bins == nb], index="shape", columns="name", values="average"
    )
    piv.plot(
        title="Benchmark digitize / onnxruntime\nn_bins=%d" % nb,
        logx=True,
        logy=True,
        ax=ax[i],
    )
