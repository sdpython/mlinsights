import distutils
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple

try:
    import numpy
except ImportError as e:
    raise ImportError(
        f"Numpy is not installed, python _executable=f{sys.executable}."
    ) from e

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel


def get_requirements(here):
    "Returns the requirements from requirements.txt."
    try:
        with open(os.path.join(here, "requirements.txt"), "r") as f:
            requirements = f.read().strip(" \n\r\t").split("\n")
    except FileNotFoundError:
        requirements = []
    if not requirements or requirements == [""]:
        requirements = ["numpy", "scipy", "scikit-learn"]
    return requirements


def get_long_description(here):
    "Returns the long description from README.rst."
    try:
        with open(os.path.join(here, "README.rst"), "r", encoding="utf-8") as f:
            long_description = "mlinsights:" + f.read().split("mlinsights:")[1]
    except FileNotFoundError:
        long_description = ""
    return long_description


def get_description():
    return (
        "More operators for onnx reference implementation and "
        "custom kernel implementation for onnxruntime."
    )


def get_version_str(here, default_version):
    VERSION_STR = default_version
    with open(os.path.join(here, "mlinsights/__init__.py"), "r") as f:
        line = [
            _
            for _ in [_.strip("\r\n ") for _ in f.readlines()]
            if _.startswith("__version__")
        ]
        if line:
            VERSION_STR = line[0].split("=")[1].strip('" ')
    if VERSION_STR is None:
        raise ValueError(f"Unable to guess the package version with here={here!r}.")
    return VERSION_STR


########################################
# C++ Helper
########################################


def find_cuda():
    try:
        p = subprocess.Popen(
            "nvidia-smi",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return False
    while True:
        output = p.stdout.readline().decode(errors="ignore")
        if output == "" and p.poll() is not None:
            break
        if output:
            if "CUDA Version:" in output:
                return True
    p.poll()
    return False


def is_windows():
    return platform.system() == "Windows"


def is_darwin():
    return platform.system() == "Darwin"


def _run_subprocess(
    args,
    cwd=None,
    capture_output=False,
    dll_path=None,
    shell=False,
    env=None,
    python_path=None,
    cuda_home=None,
    cuda_version=None,
):
    if env is None:
        env = {}
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if cuda_version is not None:
        if is_windows():
            cuda_path = (
                f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{cuda_version}"
            )
        elif is_darwin():
            cuda_path = f"/Developer/NVIDIA/CUDA-{cuda_version}"
        else:
            cuda_path = f"/usr/local/cuda-{cuda_version}/bin"
        if "PATH" in my_env:
            my_env["PATH"] = cuda_path + os.pathsep + my_env["PATH"]
        else:
            my_env["PATH"] = cuda_path

    if dll_path:
        if is_windows():
            if "PATH" in my_env:
                my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
            else:
                my_env["PATH"] = dll_path
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    if is_windows():
        py_path = os.path.dirname(sys.executable)
        if "PATH" in my_env:
            my_env["PATH"] = py_path + os.pathsep + my_env["PATH"]
        else:
            my_env["PATH"] = py_path

    # Add nvcc's folder to PATH env so that our cmake file can find nvcc
    if cuda_home:
        my_env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + my_env["PATH"]

    if python_path:
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)

    p = subprocess.Popen(
        args,
        cwd=cwd,
        shell=shell,
        env=my_env,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    raise_exception = False
    while True:
        output = p.stdout.readline().decode(errors="ignore")
        if output == "" and p.poll() is not None:
            break
        if output:
            out = output.rstrip()
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
            if (
                "fatal error" in output
                or "CMake Error" in output
                or "gmake: ***" in output
                or "): error C" in output
                or ": error: " in output
            ):
                raise_exception = True
    rc = p.poll()
    if raise_exception:
        raise RuntimeError("An error was found in the output. The build is stopped.")
    return rc


########################################
# C++ CMake Extension
########################################


class CMakeExtension(Extension):
    def __init__(self, name: str, library: str = "") -> None:
        super().__init__(name, sources=[])
        print(f"-- setup: add extension {name}")
        self.library_file = os.fspath(Path(library).resolve())


class cmake_build_class_extension(Command):
    user_options = [
        *build_ext.user_options,
        (
            "use-cuda=",
            None,
            "If cuda is available, CUDA is "
            "used by default unless this option is set to 0",
        ),
        ("use-nvtx=", None, "Enables compilation with NVTX events."),
        (
            "cuda-version=",
            None,
            "If cuda is available, it searches the installed version "
            "unless this option is defined.",
        ),
        (
            "parallel=",
            None,
            "Parallelization",
        ),
        (
            "ort-version=",
            None,
            "onnxruntime version, a path is allowed",
        ),
        (
            "cuda-build=",
            None,
            "CUDA code can be compiled to be working with "
            "different architectures, this flag can optimize "
            "for a specific machine, possible values: DEFAULT, "
            "H100, H100opt",
        ),
        (
            "cuda-link=",
            None,
            "CUDA can statically linked (STATIC) or dynamically "
            "(SHARED), default is STATIC."
            "STATIC",
        ),
        (
            "manylinux=",
            None,
            "Enforces the compilation with manylinux, default is set to 0.",
        ),
    ]

    def initialize_options(self):
        self.use_nvtx = None
        self.use_cuda = None
        self.cuda_version = None
        self.parallel = None
        self.manylinux = None
        self.cuda_build = "DEFAULT"
        self.cuda_link = "STATIC"

        self._parent.initialize_options(self)

        # boolean
        b_values = {0, 1, "1", "0", True, False}
        t_values = {1, "1", True}
        for att in ["use_nvtx", "use_cuda", "manylinux"]:
            v = getattr(self, att)
            if v is not None:
                continue
            v = os.environ.get(att.upper(), None)
            if v is None:
                continue
            if v not in b_values:
                raise ValueError(f"Unable to interpret value {v} for {att.upper()!r}.")
            print(f"-- setup: use env {att.upper()}={v in t_values}")
            setattr(self, att, v in t_values)

        if self.cuda_build is None:
            self.cuda_build = os.environ.get("CUDA_BUILD", None)
            if self.cuda_build not in ("", None):
                print(f"-- setup: use env CUDA_BUILD={self.cuda_build}")
        if self.cuda_version is None:
            self.cuda_version = os.environ.get("CUDA_VERSION", None)
            if self.cuda_version not in ("", None):
                print(f"-- setup: use env CUDA_VERSION={self.cuda_version}")
        if self.use_nvtx is None:
            self.use_nvtx = False
        if self.manylinux is None:
            self.manylinux = False

    def finalize_options(self):
        self._parent.finalize_options(self)

        b_values = {0, 1, "1", "0", True, False, "True", "False"}
        if self.use_nvtx not in b_values:
            raise ValueError(f"use_nvtx={self.use_nvtx!r} must be in {b_values}.")
        if self.use_cuda is None:
            self.use_cuda = find_cuda()
        if self.use_cuda not in b_values:
            raise ValueError(f"use_cuda={self.use_cuda!r} must be in {b_values}.")

        self.use_nvtx = self.use_nvtx in {1, "1", True, "True"}
        self.use_cuda = self.use_cuda in {1, "1", True, "True"}
        self.manylinux = self.manylinux in {1, "1", True, "True"}

        if self.cuda_version in (None, ""):
            self.cuda_version = None
        build = {"DEFAULT", "H100", "H100opt"}
        if self.cuda_build not in build:
            raise ValueError(f"cuda-build={self.cuda_build!r} not in {build}.")
        link = {"STATIC", "SHARED"}
        if self.cuda_link not in link:
            raise ValueError(f"cuda-link={self.cuda_link!r} not in {link}.")

        options = {o[0]: o for o in self.user_options}
        keys = list(sorted(options.keys()))
        for na in keys:
            opt = options[na]
            name = opt[0].replace("-", "_").strip("=")
            print(f"-- setup: option {name}={getattr(self, name, None)}")

    def get_cmake_args(self, cfg: str) -> List[str]:
        """
        Returns the argument for cmake.

        :param cfg: configuration (Release, ...)
        :return: build_path, self.build_lib
        """
        iswin = is_windows()
        isdar = is_darwin()
        cmake_cmd_args = []

        path = sys.executable
        vers = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        versmm = f"{sys.version_info.major}.{sys.version_info.minor}"
        module_ext = distutils.sysconfig.get_config_var("EXT_SUFFIX")

        here = os.path.dirname(__file__)
        if here == "":
            here = "."

        manylinux_tags = [
            "manylinux1_x86_64",
            "manylinux1_i686",
            "manylinux2010_x86_64",
            "manylinux2010_i686",
            "manylinux2014_x86_64",
            "manylinux2014_i686",
            "manylinux2014_aarch64",
            "manylinux2014_armv7l",
            "manylinux2014_ppc64",
            "manylinux2014_ppc64le",
            "manylinux2014_s390x",
            "manylinux_2_24_x86_64",
            "manylinux_2_24_aarch64",
            "manylinux_2_28_x86_64",
            "manylinux_2_28_aarch64",
            "manylinux_2_35_x86_64",
            "manylinux_2_35_aarch64",
        ]
        is_manylinux = (
            self.manylinux or os.environ.get("AUDITWHEEL_PLAT", None) in manylinux_tags
        )

        cmake_args = [
            f"-DPYTHON_EXECUTABLE={path}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPYTHON_VERSION={vers}",
            f"-DPYTHON_VERSION_MM={versmm}",
            f"-DPYTHON_MODULE_EXTENSION={module_ext}",
            f"-DMLINSIGHTS_VERSION={get_version_str(here, None)}",
            f"-DPYTHON_MANYLINUX={1 if is_manylinux else 0}",
        ]
        if self.parallel is not None:
            cmake_args.append(f"-j{self.parallel}")

        if self.use_nvtx:
            cmake_args.append("-DUSE_NVTX=1")
        cmake_args.append("-DUSE_CUDA=0")  # {1 if self.use_cuda else 0}")
        if self.use_cuda:
            cmake_args.append(f"-DCUDA_BUILD={self.cuda_build}")
            cmake_args.append(f"-DCUDA_LINK={self.cuda_link}")
        cuda_version = self.cuda_version
        if cuda_version not in (None, ""):
            cmake_args.append(f"-DCUDA_VERSION={cuda_version}")

        if iswin or isdar:
            include_dir = sysconfig.get_paths()["include"].replace("\\", "/")
            lib_dir = (
                sysconfig.get_config_var("LIBDIR")
                or sysconfig.get_paths()["stdlib"]
                or ""
            ).replace("\\", "/")
            numpy_include_dir = numpy.get_include().replace("\\", "/")
            cmake_args.extend(
                [
                    f"-DPYTHON_INCLUDE_DIR={include_dir}",
                    # f"-DPYTHON_LIBRARIES={lib_dir}",
                    f"-DPYTHON_LIBRARY_DIR={lib_dir}",
                    f"-DPYTHON_NUMPY_INCLUDE_DIR={numpy_include_dir}",
                    # "-DUSE_SETUP_PYTHON=1",
                    f"-DPYTHON_NUMPY_VERSION={numpy.__version__}",
                ]
            )
            os.environ["PYTHON_NUMPY_INCLUDE_DIR"] = numpy_include_dir

        cmake_args += cmake_cmd_args
        return cmake_args

    def build_cmake(self, cfg: str, cmake_args: List[str]) -> Tuple[str, str]:
        """
        Calls cmake.

        :param cfg: configuration (Release, ...)
        :param cmake_args: cmake aguments
        :return: build_path, self.build_lib
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        build_temp = getattr(self, "build_temp", None)
        if build_temp is None:
            build_temp = os.path.join(this_dir, "build", "install")

        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Builds the project.
        build_path = os.path.abspath(build_temp)
        build_lib = getattr(self, "build_lib", build_path)
        cmake_args = [
            *cmake_args,
            f"-DSETUP_BUILD_PATH={os.path.abspath(build_path)}",
            f"-DSETUP_BUILD_LIB={os.path.abspath(build_lib)}",
        ]
        with open(
            os.path.join(os.path.dirname(__file__), ".build_path.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(build_path)
        # build_path = os.path.join(this_dir, "build")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        source_path = os.path.join(this_dir, "_cmake")

        cmd = ["cmake", "-S", source_path, "-B", build_path, *cmake_args]
        print(f"-- setup: version={sys.version_info!r}")
        print(f"-- setup: cwd={os.getcwd()!r}")
        print(f"-- setup: source_path={source_path!r}")
        print(f"-- setup: build_path={build_path!r}")
        print(f"-- setup: cmd={' '.join(cmd)}")
        _run_subprocess(
            cmd, cwd=build_path, capture_output=True, cuda_version=self.cuda_version
        )

        # then build
        print()
        cmd = ["cmake", "--build", build_path, "--config", cfg]
        print(f"-- setup: cwd={os.getcwd()!r}")
        print(f"-- setup: build_path={build_path!r}")
        print(f"-- setup: cmd={' '.join(cmd)}")
        _run_subprocess(
            cmd, cwd=build_path, capture_output=True, cuda_version=self.cuda_version
        )
        print("-- setup: done.")
        return build_path, build_lib

    def process_extensions(self, cfg: str, build_path: str, build_lib: str):
        """
        Copies the python extensions built by cmake into python subfolders.

        :param cfg: configuration (Release, ...)
        :param build_path: where it was built
        :param build_lib: built library
        """
        if not hasattr(self, "extensions"):
            raise RuntimeError(f"Unable to get the list of extensions: {dir(self)}.")
        iswin = is_windows()
        for ext in self.extensions:
            full_name = ext._file_name
            name = os.path.split(full_name)[-1]
            if iswin:
                looks = [
                    os.path.join(build_path, cfg, full_name),
                    os.path.join(build_path, cfg, name),
                ]
            else:
                looks = [
                    os.path.join(build_path, full_name),
                    os.path.join(build_path, name),
                ]
            looks_exists = [look for look in looks if os.path.exists(look)]
            if not looks_exists:
                raise FileNotFoundError(
                    f"Unable to find {name!r} as {looks!r} (full_name={full_name!r}), "
                    f"build_path contains {os.listdir(build_path)}."
                )
            else:
                look = looks_exists[0]
            dest = os.path.join(build_lib, os.path.split(full_name)[0])
            if not os.path.exists(dest):
                os.makedirs(dest)
            if not os.path.exists(look):
                raise FileNotFoundError(f"Unable to find {look!r}.")
            if not os.path.exists(dest):
                raise FileNotFoundError(f"Unable to find folder {dest!r}.")
            print(f"-- setup: copy-2 {look!r} to {dest!r}")
            shutil.copy(look, dest)

    def run_cmake(self):
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        cfg = "Release"
        cmake_args = self.get_cmake_args(cfg)
        build_path, build_lib = self.build_cmake(cfg, cmake_args)
        if hasattr(self, "extensions"):
            print("-- process_extensions")
            self.process_extensions(cfg, build_path, build_lib)
        else:
            print("-- skip process_extensions")
        print("-- done")


class cmake_build_ext(cmake_build_class_extension, build_ext):
    _parent = build_ext
    user_options = build_ext.user_options + cmake_build_class_extension.user_options

    def run(self):
        # cmake_build_class_extension.run_cmake(self)
        return build_ext.run(self)

    def build_extensions(self):
        self.run_cmake()


class cmake_build_py(cmake_build_class_extension, build_py):
    _parent = build_py
    user_options = build_py.user_options + cmake_build_class_extension.user_options

    def run(self):
        return build_py.run(self)


class cmake_develop(cmake_build_class_extension, develop):
    _parent = develop
    user_options = develop.user_options + cmake_build_class_extension.user_options

    def run(self):
        return develop.run(self)


class cmake_install(cmake_build_class_extension, install):
    _parent = install
    user_options = install.user_options + cmake_build_class_extension.user_options

    def run(self):
        return install.run(self)


class cmake_bdist_wheel(cmake_build_class_extension, bdist_wheel):
    _parent = bdist_wheel
    user_options = bdist_wheel.user_options + cmake_build_class_extension.user_options

    def run(self):
        return bdist_wheel.run(self)


def get_ext_modules():
    if is_windows():
        ext = "pyd"
    elif is_darwin():
        ext = "dylib"
    else:
        ext = "so"

    cuda_extensions = []
    has_cuda = find_cuda()
    if has_cuda:
        add_cuda = True
        if "--use-cuda" in sys.argv:
            pos = sys.argv.index("--use-cuda")
            if len(sys.argv) > pos + 1 and sys.argv[pos + 1] in (
                "0",
                0,
                False,
                "False",
            ):
                add_cuda = False
        elif "--use-cuda=0" in sys.argv:
            add_cuda = False
        elif os.environ.get("USE_CUDA", None) in {0, "0", False}:
            add_cuda = False
        if add_cuda:
            cuda_extensions.extend([])
    elif "--with-cuda=1" in sys.argv or "--with-cuda" in sys.argv:
        raise RuntimeError(
            "CUDA is not available, it cannot be build with CUDA depsite "
            "option '--with-cuda=1'."
        )
    ext_modules = [
        CMakeExtension(
            "mlinsights.mlmodel.direct_blas_lapack",
            f"mlinsights/mlmodel/direct_blas_lapack.{ext}",
        ),
        CMakeExtension(
            "mlinsights.mlmodel._piecewise_tree_regression_common",
            f"mlinsights/mlmodel/_piecewise_tree_regression_common.{ext}",
        ),
        CMakeExtension(
            "mlinsights.mlmodel.piecewise_tree_regression_criterion",
            f"mlinsights/mlmodel/piecewise_tree_regression_criterion.{ext}",
        ),
        CMakeExtension(
            "mlinsights.mlmodel.piecewise_tree_regression_criterion_fast",
            f"mlinsights/mlmodel/piecewise_tree_regression_criterion_fast.{ext}",
        ),
        CMakeExtension(
            "mlinsights.mlmodel.piecewise_tree_regression_criterion_linear",
            f"mlinsights/mlmodel/piecewise_tree_regression_criterion_linear.{ext}",
        ),
        CMakeExtension(
            "mlinsights.mltree._tree_digitize",
            f"mlinsights/mltree/_tree_digitize.{ext}",
        ),
        *cuda_extensions,
    ]
    return ext_modules


######################
# beginning of setup
######################

here = os.path.dirname(__file__)
if here == "":
    here = "."


def get_package_data():
    known_extensions = [
        "*.cc",
        "*.cpp",
        "*.cu",
        "*.cuh",
        "*.dylib",
        "*.h",
        "*.hpp",
        "*.pyd",
        "*.pyx",
        "*.so*",
        "*.dll",
    ]
    return {
        "mlinsights": known_extensions,
        "mlinsights.mlmodel": known_extensions,
        "mlinsights.mltree": known_extensions,
    }


setup(
    name="mlinsights",
    version=get_version_str(here, "0.5.1"),
    description=get_description(),
    long_description=get_long_description(here),
    author="Xavier Dupré",
    author_email="xavier.dupre@gmail.com",
    url="https://github.com/sdpython/mlinsights",
    package_data=get_package_data(),
    setup_requires=["numpy", "scipy", "onnx"],
    install_requires=get_requirements(here),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    cmdclass={
        "build_ext": cmake_build_ext,
        "build_py": cmake_build_py,
        "bdist_wheel": cmake_bdist_wheel,
        "cmake_build": cmake_build_class_extension,
        "develop": cmake_develop,
        "install": cmake_install,
    },
    ext_modules=get_ext_modules(),
)
