import os
import unittest
import pandas
from sklearn import __version__ as sklver
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer as Imputer
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper import compare_module_version
from mlinsights.mlmodel import CategoriesToIntegers
from mlinsights.mlmodel import (
    run_test_sklearn_pickle,
    run_test_sklearn_clone,
    run_test_sklearn_grid_search_cv,
)

skipped_warnings = (ConvergenceWarning, UserWarning, FitFailedWarning)


class TestCategoriesToIntegers(ExtTestCase):
    @ignore_warnings(skipped_warnings)
    def test_categories_to_integers(self):
        data = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "data", "adult_set.txt"
        )
        df = pandas.read_csv(data, sep="\t")

        trans = CategoriesToIntegers()
        trans.fit(df)
        self.assertIsInstance(str(trans), str)
        newdf = trans.transform(df)
        exp = [
            "age",
            "final_weight",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "marital_status= Divorced",
            "marital_status= Married-AF-spouse",
            "marital_status= Married-civ-spouse",
            "marital_status= Married-spouse-absent",
            "marital_status= Never-married",
            "marital_status= Separated",
            "marital_status= Widowed",
            "sex= Female",
            "sex= Male",
            "education= 10th",
            "education= 11th",
            "education= 12th",
            "education= 1st-4th",
            "education= 5th-6th",
            "education= 7th-8th",
            "education= 9th",
            "education= Assoc-acdm",
            "education= Assoc-voc",
            "education= Bachelors",
            "education= Doctorate",
            "education= HS-grad",
            "education= Masters",
            "education= Preschool",
            "education= Prof-school",
            "education= Some-college",
            "native_country= ?",
            "native_country= Cambodia",
            "native_country= Canada",
            "native_country= China",
            "native_country= Columbia",
            "native_country= Cuba",
            "native_country= Dominican-Republic",
            "native_country= Ecuador",
            "native_country= El-Salvador",
            "native_country= England",
            "native_country= France",
            "native_country= Germany",
            "native_country= Guatemala",
            "native_country= Haiti",
            "native_country= Honduras",
            "native_country= India",
            "native_country= Iran",
            "native_country= Italy",
            "native_country= Jamaica",
            "native_country= Laos",
            "native_country= Mexico",
            "native_country= Philippines",
            "native_country= Poland",
            "native_country= Portugal",
            "native_country= Puerto-Rico",
            "native_country= South",
            "native_country= Taiwan",
            "native_country= Thailand",
            "native_country= United-States",
            "race= Amer-Indian-Eskimo",
            "race= Asian-Pac-Islander",
            "race= Black",
            "race= Other",
            "race= White",
            "relationship= Husband",
            "relationship= Not-in-family",
            "relationship= Other-relative",
            "relationship= Own-child",
            "relationship= Unmarried",
            "relationship= Wife",
            "workclass= ?",
            "workclass= Federal-gov",
            "workclass= Local-gov",
            "workclass= Private",
            "workclass= Self-emp-inc",
            "workclass= Self-emp-not-inc",
            "workclass= State-gov",
            "income= <=50K",
            "income= >50K",
            "occupation= ?",
            "occupation= Adm-clerical",
            "occupation= Armed-Forces",
            "occupation= Craft-repair",
            "occupation= Exec-managerial",
            "occupation= Farming-fishing",
            "occupation= Handlers-cleaners",
            "occupation= Machine-op-inspct",
            "occupation= Other-service",
            "occupation= Priv-house-serv",
            "occupation= Prof-specialty",
            "occupation= Protective-serv",
            "occupation= Sales",
            "occupation= Tech-support",
            "occupation= Transport-moving",
        ]
        exp.sort()
        ret = list(newdf.columns)
        ret.sort()
        self.assertEqual(len(ret), len(exp))
        self.assertEqual(exp, ret)

    @ignore_warnings(skipped_warnings)
    def test_categories_to_integers_big(self):
        data = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "data", "adult_set.txt"
        )
        df = pandas.read_csv(data, sep="\t")

        trans = CategoriesToIntegers(single=True)
        trans.fit(df)
        newdf = trans.transform(df)
        self.assertEqual(len(newdf.columns), len(df.columns))
        self.assertEqual(list(newdf.columns), list(df.columns))  # pylint: disable=E1101
        newdf2 = trans.fit_transform(df)
        self.assertEqual(newdf, newdf2)
        rep = repr(trans)
        self.assertStartsWith(
            "CategoriesToIntegers(", rep.replace(" ", "").replace("\n", "")
        )
        self.assertIn("single=True", rep.replace(" ", "").replace("\n", ""))

    @ignore_warnings(skipped_warnings)
    def test_categories_to_integers_pickle(self):
        data = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "data", "adult_set.txt"
        )
        df = pandas.read_csv(data, sep="\t")
        run_test_sklearn_pickle(lambda: CategoriesToIntegers(skip_errors=True), df)

    @ignore_warnings(skipped_warnings)
    def test_categories_to_integers_clone(self):
        self.maxDiff = None
        run_test_sklearn_clone(lambda: CategoriesToIntegers())

    @ignore_warnings(skipped_warnings)
    def test_categories_to_integers_grid_search(self):
        data = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "data", "adult_set.txt"
        )
        df = pandas.read_csv(data, sep="\t")
        X = df.drop("income", axis=1)
        y = df["income"]  # pylint: disable=E1136
        pipe = make_pipeline(CategoriesToIntegers(), LogisticRegression())
        self.assertRaise(
            lambda: run_test_sklearn_grid_search_cv(lambda: pipe, df), ValueError
        )
        if (
            compare_module_version(sklver, "0.24") >= 0
            and compare_module_version(  # pylint: disable=R1716
                pandas.__version__, "1.3"
            )
            < 0
        ):
            self.assertRaise(
                lambda: run_test_sklearn_grid_search_cv(
                    lambda: pipe, X, y, categoriestointegers__single=[True, False]
                ),
                ValueError,
                "Unable to find category value",
            )
        pipe = make_pipeline(
            CategoriesToIntegers(),
            Imputer(strategy="most_frequent"),
            LogisticRegression(n_jobs=1),
        )
        try:
            res = run_test_sklearn_grid_search_cv(
                lambda: pipe,
                X,
                y,
                categoriestointegers__single=[True, False],
                categoriestointegers__skip_errors=[True],
            )
        except AttributeError as e:
            if compare_module_version(sklver, "0.24") < 0:
                return
            raise e
        self.assertIn("model", res)
        self.assertIn("score", res)
        self.assertGreater(res["score"], 0)
        self.assertLesser(res["score"], 1)


if __name__ == "__main__":
    unittest.main()
