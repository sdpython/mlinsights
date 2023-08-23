import os
import sys
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
from mlinsights import __version__, has_cuda

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.docassert",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "mlinsights"
copyright = "2023, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True
issues_github_path = "sdpython/mlinsights"


def setup(app):
    app.add_config_value("HAS_CUDA", "1" if has_cuda() else "0", "env")


html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_static_path = ["_static"]
html_sourcelink_suffix = ""

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "mlinsights",
    (
        "https://github.com/sdpython/mlinsights/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

intersphinx_mapping = {
    "onnx": ("https://onnx.ai/onnx/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
    ("py:class", "sklearn.ensemble.RandomForestRegressor"),
    ("py:class", "unittest.case.TestCase"),
]

nitpick_ignore_regex = [
    ("py:class", ".*numpy[.].*"),
    ("py:func", ".*[.]PyCapsule[.].*"),
    ("py:func", ".*numpy[.].*"),
    ("py:func", ".*scipy[.].*"),
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

epkg_dictionary = {
    "cmake": "https://cmake.org/",
    "CPUExecutionProvider": "https://onnxruntime.ai/docs/execution-providers/",
    "cublasLtMatmul": "https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmul#cublasltmatmul",
    "CUDA": "https://developer.nvidia.com/",
    "cuda_gemm.cu": "https://github.com/sdpython/mlinsights/blob/main/mlinsights/validation/cuda/cuda_gemm.cu#L271",
    "cudnn": "https://developer.nvidia.com/cudnn",
    "CUDAExecutionProvider": "https://onnxruntime.ai/docs/execution-providers/",
    "custom_gemm.cu": "https://github.com/sdpython/mlinsights/blob/main/mlinsights/ortops/tutorial/cuda/custom_gemm.cu",
    "cython": "https://cython.org/",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "eigen": "https://eigen.tuxfamily.org/",
    "gcc": "https://gcc.gnu.org/",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "nccl": "https://developer.nvidia.com/nccl",
    "numpy": "https://numpy.org/",
    "numba": "https://numba.pydata.org/",
    "nvidia-smi": "https://developer.nvidia.com/nvidia-system-management-interface",
    "nvprof": "https://docs.nvidia.com/cuda/profiler-users-guide/index.html",
    "onnx": "https://onnx.ai/onnx/",
    "ONNX": "https://onnx.ai/",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-training": "https://github.com/microsoft/onnxruntime/tree/main/orttraining",
    "onnxruntime releases": "https://github.com/microsoft/onnxruntime/releases",
    "onnx-array-api": ("https://sdpython.github.io/doc/onnx-array-api/dev/"),
    "onnxruntime C API": "https://onnxruntime.ai/docs/api/c/",
    "onnxruntime Graph Optimizations": (
        "https://onnxruntime.ai/docs/performance/"
        "model-optimizations/graph-optimizations.html"
    ),
    "openmp": "https://www.openmp.org/",
    "protobuf": "https://github.com/protocolbuffers/protobuf",
    "pybind11": "https://github.com/pybind/pybind11",
    "pyinstrument": "https://github.com/joerick/pyinstrument",
    "python": "https://www.python.org/",
    "Python C API": "https://docs.python.org/3/c-api/index.html",
    "pytorch": "https://pytorch.org/",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "torch": "https://pytorch.org/docs/stable/torch.html",
    "tqdm": "https://tqdm.github.io/",
    "TreeEnsembleClassifier": "https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html",
    "TreeEnsembleRegressor": "https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html",
    "WSL": "https://docs.microsoft.com/en-us/windows/wsl/install",
}
