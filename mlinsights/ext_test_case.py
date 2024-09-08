import os
import sys
import unittest
import warnings
import zipfile
from io import BytesIO
from urllib.request import urlopen
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy
from numpy.testing import assert_allclose
import pandas


def unit_test_going():
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def is_azure() -> bool:
    "Tells if the job is running on Azure DevOps."
    return os.environ.get("AZURE_HTTP_USER_AGENT", "undefined") != "undefined"


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


def skipif_ci_windows(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_windows() and is_azure():
        msg = f"Test does not work on azure pipeline (Windows). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_apple(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_apple() and is_azure():
        msg = f"Test does not work on azure pipeline (Apple). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def measure_time(
    stmt: Union[str, Callable],
    context: Optional[Dict[str, Any]] = None,
    repeat: int = 10,
    number: int = 50,
    warmup: int = 1,
    div_by_number: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string or callable
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param warmup: number of iteration to do before starting the
        real measurement
    :param div_by_number: divide by the number of executions
    :param max_time: execute the statement until the total goes
        beyond this time (approximatively), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    .. runpython::
        :showcode:

        from onnx_extended.ext_test_case import measure_time
        from math import cos

        res = measure_time(lambda: cos(0.5))
        print(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(
            f"stmt is not callable or a string but is of type {type(stmt)!r}."
        )
    if context is None:
        context = {}

    if isinstance(stmt, str):
        tim = Timer(stmt, globals=context)
    else:
        tim = Timer(stmt)

    if warmup > 0:
        warmup_time = tim.timeit(warmup)
    else:
        warmup_time = 0

    if max_time is not None:
        if not div_by_number:
            raise ValueError(
                "div_by_number must be set to True of max_time is defined."
            )
        i = 1
        total_time = 0.0
        results = []
        while True:
            for j in (1, 2):
                number = i * j
                time_taken = tim.timeit(number)
                results.append((number, time_taken))
                total_time += time_taken
                if total_time >= max_time:
                    break
            if total_time >= max_time:
                break
            ratio = (max_time - total_time) / total_time
            ratio = max(ratio, 1)
            i = int(i * ratio)

        res = numpy.array(results)
        tw = res[:, 0].sum()
        ttime = res[:, 1].sum()
        mean = ttime / tw
        ave = res[:, 1] / res[:, 0]
        dev = (((ave - mean) ** 2 * res[:, 0]).sum() / tw) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(ave),
            max_exec=numpy.max(ave),
            repeat=1,
            number=tw,
            ttime=ttime,
        )
    else:
        res = numpy.array(tim.repeat(repeat=repeat, number=number))
        if div_by_number:
            res /= number

        mean = numpy.mean(res)
        dev = numpy.mean(res**2)
        dev = (dev - mean**2) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(res),
            max_exec=numpy.max(res),
            repeat=repeat,
            number=number,
            ttime=res.sum(),
        )

    if "values" in context:
        if hasattr(context["values"], "shape"):
            mes["size"] = context["values"].shape[0]
        else:
            mes["size"] = len(context["values"])
    else:
        mes["context_size"] = sys.getsizeof(context)
    mes["warmup_time"] = warmup_time
    return mes


class ExtTestCase(unittest.TestCase):
    _warns: List[type] = []

    def assertExists(self, name):
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertEqual(self, a, b):
        if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray):
            self.assertEqualArray(a, b)
        elif isinstance(a, pandas.DataFrame) and isinstance(b, pandas.DataFrame):
            self.assertEqual(list(a.columns), list(b.columns))
            self.assertEqualArray(a.values, b.values)
        else:
            try:
                super().assertEqual(a, b)
            except ValueError as e:
                raise AssertionError(
                    f"a and b are not equal, type(a)={type(a)}, type(b)={type(b)}"
                ) from e

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if expected.dtype not in (numpy.int32, numpy.int64) or value.dtype not in (
            numpy.int32,
            numpy.int64,
        ):
            self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertNotEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if expected.dtype not in (numpy.int32, numpy.int64) or value.dtype not in (
            numpy.int32,
            numpy.int64,
        ):
            self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        try:
            assert_allclose(expected, value, atol=atol, rtol=rtol)
        except AssertionError:
            return True
        raise AssertionError(f"Both arrays are equal, atol={atol}, rtol={rtol}.")

    def assertEqualDataFrame(self, d1, d2, **kwargs):
        """
        Checks that two dataframes are equal.
        Calls :epkg:`pandas:testing:assert_frame_equal`.
        """
        from pandas.testing import assert_frame_equal

        assert_frame_equal(d1, d2, **kwargs)

    def assertInAlmostEqual(
        self,
        value: float,
        expected_values: Sequence[float],
        atol: float = 0,
        rtol: float = 0,
    ):
        last_e = None
        for s in expected_values:
            try:
                self.assertAlmostEqual(value, s, atol=atol, rtol=rtol)
                return
            except AssertionError as e:
                last_e = e
        if last_e is not None:
            raise AssertionError(
                f"Value {value} not in set {expected_values}."
            ) from last_e

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(
        self, fct: Callable, exc_type: type[Exception], msg: Optional[str] = None
    ):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")  # noqa: B904
            if msg is None:
                return
            self.assertIn(msg, str(e))
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    def assertLesser(self, a, b):
        if a == b:
            return
        self.assertLess(a, b)

    def assertGreater(self, a, b):
        if a == b:
            return
        super().assertGreater(a, b)

    def assertEqualFloat(self, a, b, precision=1e-5):
        """
        Checks that ``abs(a-b) < precision``.
        """
        mi = min(abs(a), abs(b))
        if mi == 0:
            d = abs(a - b)
            try:
                self.assertLesser(d, precision)
            except AssertionError:
                raise AssertionError(f"{a} != {b} (p={precision})")  # noqa: B904
        else:
            r = float(abs(a - b)) / mi
            try:
                self.assertLesser(r, precision)
            except AssertionError:
                raise AssertionError(f"{a} != {b} (p={precision})")  # noqa: B904

    def assertEndsWith(self, suffix: str, text: str):
        if not text.endswith(suffix):
            raise AssertionError(f"Unable to find {suffix!r} in {text!r}.")

    def assertVersionGreaterOrEqual(self, v1, v2):
        from packaging.version import Version

        if Version(v1) < Version(v2):
            raise AssertionError(
                f"Unexpected version, condition {v1} >= {v2} is not true."
            )

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}", stacklevel=0)

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
            res = fct()
        return res, sout.getvalue(), serr.getvalue()


def get_parsed_args(
    name: str,
    scenarios: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    epilog: Optional[str] = None,
    number: int = 10,
    repeat: int = 10,
    warmup: int = 5,
    sleep: float = 0.1,
    tries: int = 2,
    expose: Optional[str] = None,
    **kwargs: Dict[str, Tuple[Union[int, str, float], str]],
) -> Namespace:
    """
    Returns parsed arguments for examples in this package.

    :param name: script name
    :param scenarios: list of available scenarios
    :param description: parser description
    :param epilog: text at the end of the parser
    :param number: default value for number parameter
    :param repeat: default value for repeat parameter
    :param warmup: default value for warmup parameter
    :param sleep: default value for sleep parameter
    :param expose: if empty, keeps all the parameters,
        if None, only publish kwargs contains, otherwise the list
        of parameters to publish separated by a comma
    :param kwargs: additional parameters,
        example: `n_trees=(10, "number of trees to train")`
    :return: parser
    """
    if description is None:
        description = f"Available options for {name}.py."
    if epilog is None:
        epilog = ""
    parser = ArgumentParser(prog=name, description=description, epilog=epilog)
    if expose is not None:
        to_publish = set(expose.split(",")) if not expose else set()
        if scenarios is not None:
            rows = ", ".join(f"{k}: {v}" for k, v in scenarios.items())
            parser.add_argument(
                "-s", "--scenario", help=f"Available scenarios: {rows}."
            )
        if not to_publish or "number" in to_publish:
            parser.add_argument(
                "-n",
                "--number",
                help=f"number of executions to measure, default is {number}",
                type=int,
                default=number,
            )
        if not to_publish or "repeat" in to_publish:
            parser.add_argument(
                "-r",
                "--repeat",
                help=f"number of times to repeat the measure, default is {repeat}",
                type=int,
                default=repeat,
            )
        if not to_publish or "warmup" in to_publish:
            parser.add_argument(
                "-w",
                "--warmup",
                help=f"number of times to repeat the measure, default is {warmup}",
                type=int,
                default=warmup,
            )
        if not to_publish or "sleep" in to_publish:
            parser.add_argument(
                "-S",
                "--sleep",
                help=f"sleeping time between two configurations, default is {sleep}",
                type=float,
                default=sleep,
            )
        if not to_publish or "tries" in to_publish:
            parser.add_argument(
                "-t",
                "--tries",
                help=f"number of tries for each configurations, default is {tries}",
                type=int,
                default=tries,
            )
    for k, v in kwargs.items():
        parser.add_argument(
            f"--{k}",
            help=f"{v[1]}, default is {v[0]}",
            type=type(v[0]),
            default=v[0],
        )

    return parser.parse_args()


def unzip_files(
    zipf: str,
    where_to: Optional[str] = None,
    fvalid: Optional[Callable] = None,
    fail_if_error=True,
    verbose: int = 0,
) -> List[Union[str, Tuple[str, Any]]]:
    """
    Unzips files from a zip archive.

    :param zipf: archive (or bytes or BytesIO)
    :param where_to: destination folder
        (can be None, the result is a list of tuple)
    :param fvalid: function which takes two paths
        (zip name, local name) and return True if the file
        must be unzipped, False otherwise, if None, the default answer is True
    :param fail_if_error: fails if an error is encountered
        (typically a weird character in a filename),
        otherwise a warning is thrown.
    :param verbose: display file names
    :return: list of unzipped files
    """
    if zipf.startswith("https:"):
        if where_to is None:
            raise ValueError(f"where_to must be specified for url {zipf!r}.")
        filename = zipf.split("/")[-1]
        dest_zip = os.path.join(where_to, filename)
        if not os.path.exists(dest_zip):
            if verbose:
                print(f"[unzip_files] downloads into {dest_zip!r} from {zipf!r}")
            with urlopen(zipf, timeout=10) as u:
                content = u.read()
            with open(dest_zip, "wb") as f:
                f.write(content)
        elif verbose:
            print(f"[unzip_files] already downloaded {dest_zip!r}")
        zipf = dest_zip

    if isinstance(zipf, bytes):
        zipf = BytesIO(zipf)

    try:
        with zipfile.ZipFile(zipf, "r"):
            pass
    except zipfile.BadZipFile as e:
        if isinstance(zipf, BytesIO):
            raise e
        raise OSError(f"Unable to read file '{zipf}'") from e

    files: List[Union[str, Tuple[str, Any]]] = []
    with zipfile.ZipFile(zipf, "r") as file:
        for info in file.infolist():
            if verbose > 1:
                print(f"[unzip_files] found file {info.filename!r}")
            if where_to is None:
                try:
                    content = file.read(info.filename)
                except zipfile.BadZipFile as e:
                    if fail_if_error:
                        raise zipfile.BadZipFile(
                            f"Unable to extract {info.filename!r} due to {e}"
                        ) from e
                    warnings.warn(
                        f"Unable to extract {info.filename!r} due to {e}",
                        UserWarning,
                        stacklevel=0,
                    )
                    continue
                files.append((info.filename, content))
                continue

            tos = os.path.join(where_to, info.filename)
            if not os.path.exists(tos):
                if fvalid and not fvalid(info.filename, tos):
                    if verbose > 1:
                        print(f"[unzip_files] skip file {info.filename!r}")
                    continue

                try:
                    data = file.read(info.filename)
                except zipfile.BadZipFile as e:
                    if fail_if_error:
                        raise zipfile.BadZipFile(
                            f"Unable to extract {info.filename!r} due to {e}"
                        ) from e
                    warnings.warn(
                        f"Unable to extract {info.filename!r} due to {e}",
                        UserWarning,
                        stacklevel=0,
                    )
                    continue
                if verbose > 0:
                    print(f"[unzip_files] write file {tos!r}")
                with open(tos, "wb") as f:
                    f.write(data)
                files.append(tos)
            elif not info.filename.endswith("/"):
                files.append(tos)
    return files
