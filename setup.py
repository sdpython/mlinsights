# -*- coding: utf-8 -*-
import sys
import os
import warnings
from setuptools import setup, Extension, find_packages
from pyquicksetup import read_version, read_readme, default_cmdclass

#########
# settings
#########

project_var_name = "mlinsights"
versionPython = f"{sys.version_info.major}.{sys.version_info.minor}"
path = "Lib/site-packages/" + project_var_name
readme = 'README.rst'
history = "HISTORY.rst"
requirements = None

KEYWORDS = [project_var_name, 'Xavier Dupré', 'machine learning',
            'scikit-learn']
DESCRIPTION = """Extends scikit-learn with a couple of new models, transformers, metrics, plotting."""
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering',
    'Topic :: Education',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 5 - Production/Stable'
]


#######
# data
#######

packages = find_packages()
package_dir = {k: os.path.join('.', k.replace(".", "/")) for k in packages}
package_data = {
    project_var_name + ".mlmodel": ["*.pxd", "*.pyx"],
}


def get_extensions():
    root = os.path.abspath(os.path.dirname(__file__))
    if sys.platform.startswith("win"):
        extra_compile_args = None
    else:
        extra_compile_args = ['-std=c++11']

    ext_modules = []

    # mlmodel

    import sklearn
    extensions = ["direct_blas_lapack"]
    spl = sklearn.__version__.split('.')
    vskl = (int(spl[0]), int(spl[1]))
    if vskl > (1, 2):
        extensions.append(("_piecewise_tree_regression_common",
                           "_piecewise_tree_regression_common120"))
    else:
        raise ImportError("Cannot build mlisinghts for scikit-learn<=1.2.")

    extensions.extend([
        "piecewise_tree_regression_criterion",
        "piecewise_tree_regression_criterion_linear",
        "piecewise_tree_regression_criterion_fast",
        "_tree_digitize",
    ])

    pattern1 = "mlinsights.%s.%s"
    import numpy
    for name in extensions:
        folder = "mltree" if name == "_tree_digitize" else "mlmodel"
        if isinstance(name, tuple):
            m = Extension(pattern1 % (folder, name[0]),
                          [f'mlinsights/{folder}/{name[1]}.pyx'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=["-O3"],
                          language='c')
        else:
            m = Extension(pattern1 % (folder, name),
                          [f'mlinsights/{folder}/{name}.pyx'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=["-O3"],
                          language='c')
        ext_modules.append(m)

    # cythonize
    from Cython.Build import cythonize
    opts = dict(boundscheck=False, cdivision=True,
                wraparound=False, language_level=3,
                cdivision_warnings=False, embedsignature=True,
                initializedcheck=False)
    ext_modules = cythonize(ext_modules, compiler_directives=opts)
    return ext_modules


try:
    ext_modules = get_extensions()
except ImportError as e:
    warnings.warn(
        f"Unable to build C++ extension with missing dependencies {e!r}.")
    ext_modules = None

# setup

setup(
    name=project_var_name,
    version=read_version(__file__, project_var_name),
    author='Xavier Dupré',
    author_email='xavier.dupre@gmail.com',
    license="MIT",
    url=f"http://www.xavierdupre.fr/app/{project_var_name}/helpsphinx/index.html",
    download_url=f"https://github.com/sdpython/{project_var_name}/",
    description=DESCRIPTION,
    long_description=read_readme(__file__),
    cmdclass=default_cmdclass(),
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    setup_requires=["pyquicksetup", 'cython', 'scipy', 'scikit-learn'],
    install_requires=['cython', 'scikit-learn>=1.3', 'pandas', 'scipy',
                      'matplotlib', 'pandas_streaming', 'numpy>=1.21'],
    ext_modules=ext_modules,  # cythonize(ext_modules),
)
