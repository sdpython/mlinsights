# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
from setuptools import setup, Extension
from setuptools import find_packages

#########
# settings
#########

project_var_name = "mlinsights"
sversion = "0.1"
versionPython = "%s.%s" % (sys.version_info.major, sys.version_info.minor)
path = "Lib/site-packages/" + project_var_name
readme = 'README.rst'
history = "HISTORY.rst"
requirements = None

KEYWORDS = project_var_name + ', Xavier Dupré'
DESCRIPTION = """Look for insights about machine learned models"""
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

packages = find_packages('src', exclude='src')
package_dir = {k: "src/" + k.replace(".", "/") for k in packages}
package_data = {}

############
# functions
############


def ask_help():
    return "--help" in sys.argv or "--help-commands" in sys.argv


def import_pyquickhelper():
    try:
        import pyquickhelper
    except ImportError:
        sys.path.append(
            os.path.normpath(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "pyquickhelper",
                        "src"))))
        try:
            import pyquickhelper
        except ImportError as e:
            message = "Module pyquickhelper is needed to build the documentation ({0}), not found in path {1} - current {2}".format(
                sys.executable,
                sys.path[-1],
                os.getcwd())
            raise ImportError(message) from e
    return pyquickhelper


def is_local():
    file = os.path.abspath(__file__).replace("\\", "/").lower()
    if "/temp/" in file and "pip-" in file:
        return False
    import_pyquickhelper()
    from pyquickhelper.pycode.setup_helper import available_commands_list
    return available_commands_list(sys.argv)


def verbose():
    print("---------------------------------")
    print("package_dir =", package_dir)
    print("packages    =", packages)
    print("package_data=", package_data)
    print("current     =", os.path.abspath(os.getcwd()))
    print("---------------------------------")

##########
# version
##########


if is_local() and not ask_help():
    def write_version():
        pyquickhelper = import_pyquickhelper()
        from pyquickhelper.pycode import write_version_for_setup
        return write_version_for_setup(__file__)

    if sys.version_info[0] != 2:
        write_version()

    versiontxt = os.path.join(os.path.dirname(__file__), "version.txt")
    if os.path.exists(versiontxt):
        with open(versiontxt, "r") as f:
            lines = f.readlines()
        subversion = "." + lines[0].strip("\r\n ")
        if subversion == ".0":
            raise Exception("Subversion is wrong: '{0}'.".format(subversion))
    else:
        raise FileNotFoundError(versiontxt)
else:
    # when the module is installed, no commit number is displayed
    subversion = ""

if "upload" in sys.argv and not subversion and not ask_help():
    # avoid uploading with a wrong subversion number
    try:
        import pyquickhelper
        pyq = True
    except ImportError:
        pyq = False
    raise Exception(
        "subversion is empty, cannot upload, is_local()={0}, pyquickhelper={1}".format(is_local(), pyq))

##############
# common part
##############

if os.path.exists(readme):
    if sys.version_info[0] == 2:
        from codecs import open
    with open(readme, "r", encoding='utf-8-sig') as f:
        long_description = f.read()
else:
    long_description = ""
if os.path.exists(history):
    if sys.version_info[0] == 2:
        from codecs import open
    with open(history, "r", encoding='utf-8-sig') as f:
        long_description += f.read()

if "--verbose" in sys.argv:
    verbose()

if is_local():
    pyquickhelper = import_pyquickhelper()
    logging_function = pyquickhelper.get_fLOG()
    logging_function(OutputPrint=True)
    must_build, run_build_ext = pyquickhelper.get_insetup_functions()

    if must_build():
        out = run_build_ext(__file__)
        print(out)

    if "build_sphinx" in sys.argv and not sys.platform.startswith("win"):
        # There is an issue with matplotlib and notebook gallery on linux
        # _tkinter.TclError: no display name and no $DISPLAY environment variable
        import matplotlib
        matplotlib.use('agg')

    from pyquickhelper.pycode import process_standard_options_for_setup
    r = process_standard_options_for_setup(
        sys.argv, __file__, project_var_name,
        unittest_modules=["pyquickhelper"],
        additional_notebook_path=["pyquickhelper",
                                  "jyquickhelper", "pandas_streaming"],
        additional_local_path=["pyquickhelper",
                               "jyquickhelper", "pandas_streaming"],
        requirements=["pyquickhelper", "jyquickhelper", "pandas_streaming"],
        layout=["html"],
        add_htmlhelp=sys.platform.startswith("win"),
        coverage_options=dict(omit=["*exclude*.py"]),
        fLOG=logging_function, github_owner='sdpython',
        covtoken=("1ac0b95d-6722-4f29-804a-e4e0d5295497", "'_UT_36_std' in outfile"))
    if not r and not ({"bdist_msi", "sdist",
                       "bdist_wheel", "publish", "publish_doc", "register",
                       "upload_docs", "bdist_wininst", "build_ext"} & set(sys.argv)):
        raise Exception("unable to interpret command line: " + str(sys.argv))
else:
    r = False

if ask_help():
    pyquickhelper = import_pyquickhelper()
    from pyquickhelper.pycode import process_standard_options_for_setup_help
    process_standard_options_for_setup_help(sys.argv)

if not r:
    if len(sys.argv) in (1, 2) and sys.argv[-1] in ("--help-commands",):
        pyquickhelper = import_pyquickhelper()
        from pyquickhelper.pycode import process_standard_options_for_setup_help
        process_standard_options_for_setup_help(sys.argv)
    else:
        pyquickhelper = import_pyquickhelper()
    from pyquickhelper.pycode import clean_readme
    long_description = clean_readme(long_description)
    root = os.path.abspath(os.path.dirname(__file__))
    if sys.platform.startswith("win"):
        extra_compile_args = None
    else:
        extra_compile_args = ['-std=c++11']
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
        package_data=package_data,
        # data_files=data_files,
        install_requires=['scikit-learn', 'pandas',
                          'pillow', 'matplotlib', 'h5py',
                          'pandas_streaming'],
        # include_package_data=True,
    )
