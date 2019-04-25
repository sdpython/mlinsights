# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
from setuptools import setup, Extension
from setuptools import find_packages
from Cython.Build import cythonize
import numpy

#########
# settings
#########

project_var_name = "mlinsights"
versionPython = "%s.%s" % (sys.version_info.major, sys.version_info.minor)
path = "Lib/site-packages/" + project_var_name
readme = 'README.rst'
history = "HISTORY.rst"
requirements = None

KEYWORDS = project_var_name + ', Xavier Dupré'
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

here = os.path.dirname(__file__)
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}

############
# functions
############


def ask_help():
    return "--help" in sys.argv or "--help-commands" in sys.argv


def is_local():
    file = os.path.abspath(__file__).replace("\\", "/").lower()
    if "/temp/" in file and "pip-" in file:
        return False
    from pyquickhelper.pycode.setup_helper import available_commands_list
    return available_commands_list(sys.argv)


def verbose():
    print("---------------------------------")
    print("package_dir =", package_dir)
    print("packages    =", packages)
    print("current     =", os.path.abspath(os.getcwd()))
    print("---------------------------------")

##########
# version
##########


if is_local() and not ask_help():
    def write_version():
        from pyquickhelper.pycode import write_version_for_setup
        return write_version_for_setup(__file__)

    write_version()

    versiontxt = os.path.join(os.path.dirname(__file__), "version.txt")
    if os.path.exists(versiontxt):
        with open(versiontxt, "r") as f:
            lines = f.readlines()
        subversion = "." + lines[0].strip("\r\n ")
        if subversion == ".0":
            raise Exception("Git version is wrong: '{0}'.".format(subversion))
    else:
        raise FileNotFoundError(versiontxt)
else:
    # when the module is installed, no commit number is displayed
    subversion = ""

if "upload" in sys.argv and not subversion and not ask_help():
    # avoid uploading with a wrong subversion number
    raise Exception(
        "Git version is empty, cannot upload, is_local()={0}".format(is_local()))

##############
# common part
##############

if os.path.exists(readme):
    with open(readme, "r", encoding='utf-8-sig') as f:
        long_description = f.read()
else:
    long_description = ""
if os.path.exists(history):
    with open(history, "r", encoding='utf-8-sig') as f:
        long_description += f.read()

if "--verbose" in sys.argv:
    verbose()

build_commmands = {"bdist_msi", "sdist",
                   "bdist_wheel", "publish", "publish_doc", "register",
                   "upload_docs", "bdist_wininst", "build_ext"}

if is_local():
    import pyquickhelper
    logging_function = pyquickhelper.get_fLOG()
    logging_function(OutputPrint=True)
    must_build, run_build_ext = pyquickhelper.get_insetup_functions()

    if must_build() and not ask_help():
        out = run_build_ext(__file__)
        print(out)

    from pyquickhelper.pycode import process_standard_options_for_setup
    r = process_standard_options_for_setup(
        sys.argv, __file__, project_var_name,
        unittest_modules=["pyquickhelper"],
        additional_notebook_path=["pyquickhelper", "cpyquickhelper",
                                  "jyquickhelper", "pandas_streaming"],
        additional_local_path=["pyquickhelper", "cpyquickhelper",
                               "jyquickhelper", "pandas_streaming"],
        requirements=["pyquickhelper", "jyquickhelper", "pandas_streaming"],
        layout=["html"],
        add_htmlhelp=sys.platform.startswith("win"),
        coverage_options=dict(omit=["*exclude*.py"]),
        fLOG=logging_function, github_owner='sdpython',
        covtoken=("1ac0b95d-6722-4f29-804a-e4e0d5295497", "'_UT_37_std' in outfile"))
    if not r and not (build_commmands & set(sys.argv)):
        raise Exception("unable to interpret command line: " + str(sys.argv))
else:
    r = False

if r and ask_help():
    from pyquickhelper.pycode import process_standard_options_for_setup_help
    process_standard_options_for_setup_help(sys.argv)

if not r:
    if len(sys.argv) in (1, 2) and sys.argv[-1] in ("--help-commands",):
        from pyquickhelper.pycode import process_standard_options_for_setup_help
        process_standard_options_for_setup_help(sys.argv)
    from pyquickhelper.pycode import clean_readme
    from mlinsights import __version__ as sversion
    long_description = clean_readme(long_description)
    root = os.path.abspath(os.path.dirname(__file__))
    if sys.platform.startswith("win"):
        extra_compile_args = None
    else:
        extra_compile_args = ['-std=c++11']

    from pyquickhelper.texthelper import compare_module_version
    import sklearn

    ext_modules = []

    # mlmodel

    extensions = ["direct_blas_lapack"]
    if compare_module_version(sklearn.__version__, "0.21") >= 0:
        extensions.extend([
            "_piecewise_tree_regression_common",
            "piecewise_tree_regression_criterion",
            "piecewise_tree_regression_criterion_linear",
            "piecewise_tree_regression_criterion_fast",
        ])
    else:
        if verbose:
            print("Cannot build all cython extensions or upgrade scikit-learn to 0.21.")

    pattern1 = "mlinsights.mlmodel.%s"
    for name in extensions:
        m = Extension(pattern1 % name,
                      ['mlinsights/mlmodel/%s.pyx' % name],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=["-O3"],
                      language='c')
        ext_modules.append(m)

    # cythonize

    opts = dict(boundscheck=False, cdivision=True,
                wraparound=False, language_level=3,
                cdivision_warnings=True)
    ext_modules = cythonize(ext_modules, compiler_directives=opts)

    # setup

    setup(
        name=project_var_name,
        version='%s%s' % (sversion, subversion),
        author='Xavier Dupré',
        author_email='xavier.dupre@gmail.com',
        license="MIT",
        url="http://www.xavierdupre.fr/app/%s/helpsphinx/index.html" % project_var_name,
        download_url="https://github.com/sdpython/%s/" % project_var_name,
        description=DESCRIPTION,
        long_description=long_description,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=packages,
        package_dir=package_dir,
        setup_requires=["pyquickhelper"],
        install_requires=['Cython', 'scikit-learn', 'pandas',
                          'matplotlib', 'pandas_streaming'],
        ext_modules=ext_modules,  # cythonize(ext_modules),
    )
