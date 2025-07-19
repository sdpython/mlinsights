===========
Change Logs
===========

0.5.2
=====

* :pr:`136`: adds Python 3.13 to CI, updates the package to support scikit-learn==1.7.1

0.5.1
=====

* :pr:`130` numpy 2.0
* :pr:`132` builds against scikit-learn==1.5.0, python 3.12

0.5.0
=====

* :pr:`118` major refactoring, changes CI, builds against scikit-learn 1.3 
* :pr:`115` Updates tree decision criterion for scikit-learn 1.2 (2023-07-02)
* :pr:`113` Removes normalize attributes (deprecated) (2022-11-29)
* :pr:`110` Fixes perplexity issue with PredictableTSNE (2022-08-06)
* :pr:`109` Use f strings in more places (2022-07-22)

0.3.649 - 2022-07-22 - 2.35Mb
=============================

* :pr:`105` Update for python 3.10 (2022-07-22)
* :pr:`108` Uses f strings (2022-07-19)

0.3.631 - 2022-05-19 - 2.21Mb
=============================

* :pr:`107` Updates CI for scikit-learn==1.1 (2022-05-18)
* :pr:`106` Fixes failing import _joblib_parallel_args (2022-02-18)
* :pr:`99` LICENSE file missing in PyPI release (2021-11-20)

0.3.614 - 2021-10-02 - 1.73Mb
=============================

* :pr:`103` Updates for scikit-learn>=1.0 (2021-10-02)
* :pr:`94` Fixed Numpy boolean array indexing issue for 2dim arrays. (2021-09-27)

0.3.606 - 2021-08-22 - 2.35Mb
=============================

* :pr:`102` Implements numpy.digitalize with a DecisionTreeRegressor (2021-08-22)
* :pr:`101` Update CI to build manylinux for python 3.9 (2021-08-18)
* :pr:`100` Support parameter positive for QuantileLinearRegression (2021-06-23)
* :pr:`96` Fixes #95, PiecewiseRegressor, makes sure target are vectors (2021-05-27)
* :pr:`95` _apply_prediction_method boolean indexing incompatible with standard sklearn format (2021-05-27)
* :pr:`80` Piecewise Estimator` binner not a decision tree (2021-05-06)
* :pr:`72` Optimal decission tree for piecewise estimator (2021-05-06)
* :pr:`98` Fixes #97, fix issue with deepcopy and criterion (2021-05-03)
* :pr:`97` piecewise_decision_tree does not compile with the latest version of scikit-learn (2021-05-03)
* :pr:`85` Fixes #70, implements DecisionTreeLogisticRegression (2021-05-02)
* :pr:`93` Include build wheel for all platforms in CI (2021-01-09)
* :pr:`89` Install fails` ModuleNotFoundError` No module named 'sklearn' (2021-01-03)
* :pr:`92` QuantileMLPRegressor does not work with scikit-learn 0.24 (2021-01-01)
* :pr:`91` Fixes regression criterion for scikit-learn 0.24 (2021-01-01)
* :pr:`90` Fixes PipelineCache for scikit-learn 0.24 (2021-01-01)
* :pr:`88` Change for scikit-learn 0.24 (2020-09-02)
* :pr:`87` Set up CI with Azure Pipelines (2020-09-02)
* :pr:`86` Update CI, use python 3.8 (2020-09-02)
* :pr:`71` update kmeans l1 to the latest kmeans (signatures changed) (2020-08-31)
* :pr:`84` style (2020-08-30)
* :pr:`83` Upgrade version (2020-08-06)
* :pr:`82` Fixes #81, skl 0.22, 0.23 together (2020-08-06)
* :pr:`81` Make mlinsights work with scikit-learn 0.22 and 0.23 (2020-08-06)
* :pr:`79` pipeline2dot fails with 'passthrough' (2020-07-16)
* :pr:`78` Removes strong dependency on pyquickhelper (2020-06-29)
* :pr:`77` Add parameter trainable to TransferTransformer (2020-06-07)
* :pr:`76` ConstraintKMeans does not produce convex clusters. (2020-06-03)
* :pr:`75` Moves kmeans with constraint from papierstat. (2020-05-27)
* :pr:`74` Fix PipelineCache after as scikti-learn 0.23 changed the way parameters is handle in pipelines (2020-05-15)
* :pr:`73` ClassifierKMeans.__repr__ fails with scikit-learn 0.23 (2020-05-14)
* :pr:`69` Optimizes k-means with norm L1 (2020-01-13)
* :pr:`66` Fix visualisation graph` does not work when column index is an integer in ColumnTransformer (2019-09-15)
* :pr:`59` Add GaussianProcesses to the notebook about confidence interval and regression (2019-09-15)
* :pr:`65` Implements a TargetTransformClassifier similar to TargetTransformRegressor (2019-08-24)
* :pr:`64` Implements a different version of TargetTransformRegressor which includes predefined functions (2019-08-24)
* :pr:`63` Add a transform which transform the target and applies the inverse function of the prediction before scoring (2019-08-24)
* :pr:`49` fix menu in documentation (2019-08-24)
* :pr:`61` Fix bug in pipeline2dot when keyword "passthrough is used" (2019-07-11)
* :pr:`60` Fix visualisation of pipeline which contains string "passthrough" (2019-07-09)
* :pr:`58` Explores a way to compute recommandations without training (2019-06-05)
* :pr:`56` Fixes #55, explore caching for scikit-learn pipeline (2019-05-22)
* :pr:`55` Explore caching for gridsearchCV (2019-05-22)
* :pr:`53` implements a function to extract intermediate model outputs within a pipeline (2019-05-07)
* :pr:`51` Implements a tfidfvectorizer which keeps more information about n-grams (2019-04-26)
* :pr:`46` implements a way to determine close leaves in a decision tree (2019-04-01)
* :pr:`44` implements a model which produces confidence intervals based on bootstrapping (2019-03-29)
* :pr:`40` implements a custom criterion for a decision tree optimizing for a linear regression (2019-03-28)
* :pr:`39` implements a custom criterion for decision tree (2019-03-26)
* :pr:`41` implements a direct call to a lapack function from cython (2019-03-25)
* :pr:`38` better implementation of a regression criterion (2019-03-25)
* :pr:`37` implements interaction_only for polynomial features (2019-02-26)
* :pr:`36` add parameter include_bias to extended features (2019-02-25)
* :pr:`34` rename PiecewiseLinearRegression into PiecewiseRegression (2019-02-23)
* :pr:`33` implement the piecewise classifier (2019-02-23)
* :pr:`31` uses joblib for piecewise linear regression (2019-02-23)
* :pr:`30` explore transpose matrix before computing the polynomial features (2019-02-17)
* :pr:`29` explore different implementation of polynomialfeatures (2019-02-15)
* :pr:`28` implement PiecewiseLinearRegression (2019-02-10)
* :pr:`27` implement TransferTransformer (2019-02-04)
* :pr:`26` add function to convert a scikit-learn pipeline into a graph (2019-02-01)
* :pr:`25` implements kind of trainable t-SNE (2019-01-31)
* :pr:`6` use keras and pytorch (2019-01-03)
* :pr:`22` modifies plot gallery to impose coordinates (2018-11-10)
* :pr:`20` implements a QuantileMLPRegressor (quantile regression with MLP) (2018-10-22)
* :pr:`19` fix issues introduced with changes in keras 2.2.4 (2018-10-06)
* :pr:`18` remove warning from scikit-learn about cloning (2018-09-16)
* :pr:`16` move CI to python 3.7 (2018-08-21)
* :pr:`17` replace as_matrix by values (pandas deprecated warning) (2018-07-29)
* :pr:`14` add transform to convert a learner into a transform (sometimes called a  featurizer) (2018-06-19)
* :pr:`13` add transform to do model stacking (2018-06-19)
* :pr:`8` move items from papierstat (2018-06-19)
* :pr:`12` fix bug in quantile regression` wrong weight for linear regression (2018-06-16)
* :pr:`11` specifying quantile (2018-06-16)
* :pr:`4` add function to compute non linear correlations (2018-06-16)
* :pr:`10` implements combination between logistic regression and k-means (2018-05-27)
* :pr:`9` move items from ensae_teaching_cs (2018-05-08)
* :pr:`7` add quantile regression (2018-05-07)
* :pr:`5` replace flake8 by code style (2018-04-14)
* :pr:`1` change background for cells in notebooks converted into rst then in html, highlight-ipython3 (2018-01-05)
* :pr:`2` save features and metadatas for the search engine and retrieves them (2017-12-03)
