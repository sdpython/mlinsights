"""
@brief      test log(time=2s)
"""

import sys
import os
import unittest
import pandas


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_

from src.mlinsights.mlmodel import CategoriesToIntegers


class TestCategoriesToIntegers(unittest.TestCase):

    def test_categories_to_integers(self):
        data = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), "data", "adult_set.txt")
        df = pandas.read_csv(data, sep="\t")

        trans = CategoriesToIntegers()
        trans.fit(df)
        newdf = trans.transform(df)
        exp = ['age', 'final_weight', 'education_num', 'capital_gain', 'capital_loss',
               'hours_per_week', 'marital_status= Divorced',
               'marital_status= Married-AF-spouse',
               'marital_status= Married-civ-spouse',
               'marital_status= Married-spouse-absent',
               'marital_status= Never-married', 'marital_status= Separated',
               'marital_status= Widowed', 'sex= Female', 'sex= Male',
               'education= 10th', 'education= 11th', 'education= 12th',
               'education= 1st-4th', 'education= 5th-6th', 'education= 7th-8th',
               'education= 9th', 'education= Assoc-acdm', 'education= Assoc-voc',
               'education= Bachelors', 'education= Doctorate', 'education= HS-grad',
               'education= Masters', 'education= Preschool', 'education= Prof-school',
               'education= Some-college', 'native_country= ?',
               'native_country= Cambodia', 'native_country= Canada',
               'native_country= China', 'native_country= Columbia',
               'native_country= Cuba', 'native_country= Dominican-Republic',
               'native_country= Ecuador', 'native_country= El-Salvador',
               'native_country= England', 'native_country= France',
               'native_country= Germany', 'native_country= Guatemala',
               'native_country= Haiti', 'native_country= Honduras',
               'native_country= India', 'native_country= Iran',
               'native_country= Italy', 'native_country= Jamaica',
               'native_country= Laos', 'native_country= Mexico',
               'native_country= Philippines', 'native_country= Poland',
               'native_country= Portugal', 'native_country= Puerto-Rico',
               'native_country= South', 'native_country= Taiwan',
               'native_country= Thailand', 'native_country= United-States',
               'race= Amer-Indian-Eskimo', 'race= Asian-Pac-Islander', 'race= Black',
               'race= Other', 'race= White', 'relationship= Husband',
               'relationship= Not-in-family', 'relationship= Other-relative',
               'relationship= Own-child', 'relationship= Unmarried',
               'relationship= Wife', 'workclass= ?', 'workclass= Federal-gov',
               'workclass= Local-gov', 'workclass= Private', 'workclass= Self-emp-inc',
               'workclass= Self-emp-not-inc', 'workclass= State-gov', 'income= <=50K',
               'income= >50K', 'occupation= ?', 'occupation= Adm-clerical',
               'occupation= Armed-Forces', 'occupation= Craft-repair',
               'occupation= Exec-managerial', 'occupation= Farming-fishing',
               'occupation= Handlers-cleaners', 'occupation= Machine-op-inspct',
               'occupation= Other-service', 'occupation= Priv-house-serv',
               'occupation= Prof-specialty', 'occupation= Protective-serv',
               'occupation= Sales', 'occupation= Tech-support',
               'occupation= Transport-moving']
        exp.sort()
        ret = list(newdf.columns)
        ret.sort()
        self.assertEqual(len(ret), len(exp))
        self.assertEqual(exp, ret)

    def test_categories_to_integers_big(self):
        data = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), "data", "adult_set.txt")
        df = pandas.read_csv(data, sep="\t")

        trans = CategoriesToIntegers(single=True)
        trans.fit(df)
        newdf = trans.transform(df)
        self.assertEqual(len(newdf.columns), len(df.columns))
        self.assertEqual(list(newdf.columns), list(df.columns))


if __name__ == "__main__":
    unittest.main()
