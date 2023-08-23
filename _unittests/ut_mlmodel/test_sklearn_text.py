# -*- coding: utf-8 -*-
import unittest
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyquickhelper.pycode import ExtTestCase
from mlinsights.mlmodel.sklearn_text import (
    TraceableTfidfVectorizer,
    TraceableCountVectorizer,
)


class TestSklearnText(ExtTestCase):
    def test_count_vectorizer(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
            mod1 = CountVectorizer(ngram_range=ng)
            mod1.fit(corpus)

            mod2 = TraceableCountVectorizer(ngram_range=ng)
            mod2.fit(corpus)

            pred1 = mod1.transform(corpus)
            pred2 = mod2.transform(corpus)
            self.assertEqualArray(pred1.todense(), pred2.todense())

            voc = mod2.vocabulary_
            for k in voc:
                self.assertIsInstance(k, tuple)

    def test_count_vectorizer_regex(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for pattern in ["[a-zA-Z ]{1,4}", "[a-zA-Z]{1,4}"]:
            for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
                mod1 = CountVectorizer(ngram_range=ng, token_pattern=pattern)
                mod1.fit(corpus)

                mod2 = TraceableCountVectorizer(ngram_range=ng, token_pattern=pattern)
                mod2.fit(corpus)

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)
                self.assertEqualArray(pred1.todense(), pred2.todense())

                voc = mod2.vocabulary_
                for k in voc:
                    self.assertIsInstance(k, tuple)
                if " ]" in pattern:
                    spaces = 0
                    for k in voc:
                        self.assertIsInstance(k, tuple)
                        for i in k:
                            if " " in i:
                                spaces += 1
                    self.assertGreater(spaces, 1)

    def test_tfidf_vectorizer(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
            mod1 = TfidfVectorizer(ngram_range=ng)
            mod1.fit(corpus)

            mod2 = TraceableTfidfVectorizer(ngram_range=ng)
            mod2.fit(corpus)

            pred1 = mod1.transform(corpus)
            pred2 = mod2.transform(corpus)
            self.assertEqualArray(pred1.todense(), pred2.todense())

            voc = mod2.vocabulary_
            for k in voc:
                self.assertIsInstance(k, tuple)

    def test_tfidf_vectorizer_regex(self):
        corpus = numpy.array(
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
                "",
            ]
        ).reshape((5,))

        for pattern in ["[a-zA-Z ]{1,4}", "[a-zA-Z]{1,4}"]:
            for ng in [(1, 1), (1, 2), (2, 2), (1, 3)]:
                mod1 = TfidfVectorizer(ngram_range=ng, token_pattern=pattern)
                mod1.fit(corpus)

                mod2 = TraceableTfidfVectorizer(ngram_range=ng, token_pattern=pattern)
                mod2.fit(corpus)

                pred1 = mod1.transform(corpus)
                pred2 = mod2.transform(corpus)

                if " ]" in pattern:
                    voc = mod2.vocabulary_
                    spaces = 0
                    for k in voc:
                        self.assertIsInstance(k, tuple)
                        for i in k:
                            if " " in i:
                                spaces += 1
                    self.assertGreater(spaces, 1)
                self.assertEqualArray(pred1.todense(), pred2.todense())


if __name__ == "__main__":
    unittest.main()
