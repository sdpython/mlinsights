"""
Traceable n-grams with tf-idf
=============================

The notebook looks into the way n-grams are stored in
`CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
and
`TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer>`_
and how the current storage (<= 0.21) is ambiguous in some cases.

Example with CountVectorizer
----------------------------

scikit-learn version
~~~~~~~~~~~~~~~~~~~~
"""

import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from mlinsights.mlmodel.sklearn_text import (
    TraceableCountVectorizer,
    TraceableTfidfVectorizer,
)


corpus = numpy.array(
    [
        "This is the first document.",
        "This document is the second document.",
        "Is this the first document?",
        "",
    ]
).reshape((4,))

mod1 = CountVectorizer(ngram_range=(1, 2))
mod1.fit(corpus)
########################################
#

mod1.transform(corpus).todense()

########################################
#


mod1.vocabulary_

########################################
#


corpus = numpy.array(
    [
        "This is the first document.",
        "This document is the second document.",
        "Is this the first document?",
        "",
    ]
).reshape((4,))

########################################
#


mod2 = TraceableCountVectorizer(ngram_range=(1, 2))
mod2.fit(corpus)
########################################
#

mod2.transform(corpus).todense()

########################################
#

mod2.vocabulary_


######################################################################
# The new class does the exact same thing but keeps n-grams in a more
# explicit form. The original form as a string is sometimes ambiguous as
# next example shows.
#
# Funny example with TfidfVectorizer
# ----------------------------------
#
# scikit-learn version
# ~~~~~~~~~~~~~~~~~~~~


corpus = numpy.array(
    [
        "This is the first document.",
        "This document is the second document.",
        "Is this the first document?",
        "",
    ]
).reshape((4,))

########################################
#

mod1 = TfidfVectorizer(ngram_range=(1, 2), token_pattern="[a-zA-Z ]{1,4}")
mod1.fit(corpus)
########################################
#

mod1.transform(corpus).todense()

########################################
#

mod1.vocabulary_


######################################################################
# mlinsights version
# ~~~~~~~~~~~~~~~~~~


mod2 = TraceableTfidfVectorizer(ngram_range=(1, 2), token_pattern="[a-zA-Z ]{1,4}")
mod2.fit(corpus)
########################################
#

mod2.transform(corpus).todense()

########################################
#

mod2.vocabulary_


######################################################################
# As you can see, the original 30th n-grams ``'t is  the'`` is a little
# but ambiguous. It is in fact ``('t is', ' the')`` as the
# *TraceableTfidfVectorizer* lets you know. The original form could have
# been ``('t', 'is  the')``, ``('t is', '  the')``, ``('t is ', ' the')``,
# ``('t is  ', 'the')``, ``('t', 'is  ', 'the')``\ … The regular
# expression gives some insights but not some information which can be
# easily used to guess the right one.
