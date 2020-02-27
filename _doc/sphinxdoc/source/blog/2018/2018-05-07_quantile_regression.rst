
.. blogpost::
    :title: Quantile regression with scikit-learn.
    :keywords: scikit-learn, quantile regression
    :date: 2018-05-07
    :categories: machine learning

    :epkg:`scikit-learn` does not have any quantile regression.
    :epkg:`statsmodels` does have one
    `QuantReg <http://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html>`_
    but I wanted to try something I did for my teachings
    `Régression Quantile
    <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/notebooks/td_note_2017_2.html?highlight=mediane>`_
    based on `Iteratively reweighted least squares
    <https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares>`_.
    I thought it was a good case study to turn a simple algorithm into
    a learner :epkg:`scikit-learn` can reused in a pipeline.
    The notebook :ref:`quantileregressionrst` demonstrates it
    and it is implemented in
    :class:`QuantileLinearRegression <mlinsights.mlmodel.quantile_regression.QuantileLinearRegression>`.
