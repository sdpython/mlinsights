"""
@file
@brief Correlations.
"""
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import clone


def non_linear_correlations(df, model, draws=5, minmax=False):
    """
    Computes non linear correlations.

    @param      df      :epkg:`pandas:DataFrame` or
                        :epkg:`numpy:array`
    @param      model   machine learned model used to compute
                        the correlations
    @param      draws   number of tries for :epkg:`bootstrap`,
                        the correlation is the average of the results
                        obtained at each draw
    @param      minmax  if True, returns three matrices correlations, min, max,
                        only the correlation matrix if False
    @return             see parameter minmax

    `Pearson Correlations <https://fr.wikipedia.org/wiki/Corr%C3%A9lation_(statistiques)>`_
    is:

    .. math::

        cor(X_i, X_j) = \\frac{cov(X_i, Y_i)}{\\sigma(X_i)\\sigma(X_j)}

    If variables are centered, :math:`\\mathbb{E}X_i=\\mathbb{E}X_j=0`,
    it becomes:

    .. math::

        cor(X_i, X_j) = \\frac{\\mathbb{E}(X_i X_j)}{\\sqrt{\\mathbb{E}X_i^2 \\mathbb{E}X_j^2}}

    If rescaled, :math:`\\mathbb{E}X_i^2=\\mathbb{E}X_j^2=1`,
    then it becomes :math:`cor(X_i, X_j) = \\mathbb{E}(X_i X_j)`.
    Let's assume we try to find a coefficient such as
    :math:`\\alpha_{ij}` minimizes the standard deviation
    of noise :math:`\\epsilon_{ij}`:

    .. math::

        X_j = \\alpha_{ij}X_i + \\epsilon_{ij}

    It is like if coefficient :math:`\\alpha_{ij}` comes from a
    a linear regression which minimizes
    :math:`\\mathbb{E}(X_j - \\alpha_{ij}X_i)^2`.
    If variable :math:`X_i`, :math:`X_j` are centered
    and rescaled: :math:`\\alpha_{ij}^* = \\mathbb{E}(X_i X_j) = cor(X_i, X_j)`.
    We extend that definition to function :math:`f` of parameter :math:`\\omega`
    defined as: :math:`f(\\omega, X) \\rightarrow \\mathbb{R}`.
    :math:`f` is not linear anymore.
    Let's assume parameter :math:`\\omega^*` minimizes
    quantity :math:`\\min_\\omega (X_j  - f(\\omega, X_i))^2`.
    Then :math:`X_j = \\alpha_{ij} \\frac{f(\\omega^*, X_i)}{\\alpha_{ij}} + \\epsilon_{ij}`
    and we choose :math:`\\alpha_{ij}` such as
    :math:`\\mathbb{E}\\left(\\frac{f(\\omega^*, X_i)^2}{\\alpha_{ij}^2}\\right) = 1`.
    Let's define a non linear correlation bounded by :math:`f` as:

    .. math::

        cor^f(X_i, X_j) = \\sqrt{ \\mathbb{E} (f(\\omega, X_i)^2 )}

    We can verify that this value is in interval`:math:`[0,1]``.
    That also means that there is no negative correlation.
    :math:`f` is a machine learned model and most of them
    usually overfit the data. The database is split into
    two parts, one is used to train the model, the other
    one to compute the correlation. The same split are used
    for every coefficient. The returned matrix is not
    necessarily symmetric.

    .. exref::
        :title: Compute non linear correlations

        The following example compute non linear correlations
        on :epkg:`Iris` datasets based on a
        :epkg:`RandomForestRegressor` model.

        .. runpython::
            :showcode:
            :warningout: FutureWarning

            import pandas
            from sklearn import datasets
            from sklearn.ensemble import RandomForestRegressor
            from mlinsights.metrics import non_linear_correlations

            iris = datasets.load_iris()
            X = iris.data[:, :4]
            df = pandas.DataFrame(X)
            df.columns = ["X1", "X2", "X3", "X4"]
            cor = non_linear_correlations(df, RandomForestRegressor())
            print(cor)

    """

    if hasattr(df, 'iloc'):
        cor = df.corr()
        cor.iloc[:, :] = 0.
        iloc = True
        if minmax:
            mini = cor.copy()
            maxi = cor.copy()
    else:
        cor = numpy.corrcoef(df, rowvar=False)
        cor[:, :] = 0.
        iloc = False
        if minmax:
            mini = cor.copy()
            maxi = cor.copy()
    df = scale(df)

    for k in range(0, draws):
        df_train, df_test = train_test_split(df, test_size=0.5)
        for i in range(cor.shape[0]):
            xi_train = df_train[:, i:i + 1]
            xi_test = df_test[:, i:i + 1]
            for j in range(cor.shape[1]):
                xj_train = df_train[:, j:j + 1]
                xj_test = df_test[:, j:j + 1]
                if len(xj_test) == 0 or len(xi_test) == 0:
                    raise ValueError(  # pragma: no cover
                        "One column is empty i={0} j={1}.".format(i, j))
                mod = clone(model)
                try:
                    mod.fit(xi_train, xj_train.ravel())
                except Exception as e:  # pragma: no cover
                    raise ValueError(
                        "Unable to compute correlation for i={0} j={1}.".format(i, j)) from e
                v = mod.predict(xi_test)
                c = (1 - numpy.var(v - xj_test.ravel()))
                co = max(c, 0) ** 0.5
                if iloc:
                    cor.iloc[i, j] += co
                    if minmax:
                        if k == 0:
                            mini.iloc[i, j] = co
                            maxi.iloc[i, j] = co
                        else:
                            mini.iloc[i, j] = min(mini.iloc[i, j], co)
                            maxi.iloc[i, j] = max(maxi.iloc[i, j], co)
                else:
                    cor[i, j] += co
                    if minmax:
                        if k == 0:
                            mini[i, j] = co
                            maxi[i, j] = co
                        else:
                            mini[i, j] = min(mini[i, j], co)
                            maxi[i, j] = max(maxi[i, j], co)
    if minmax:
        return cor / draws, mini, maxi
    return cor / draws
