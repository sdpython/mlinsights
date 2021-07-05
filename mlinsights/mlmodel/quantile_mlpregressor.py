# -*- coding: utf-8 -*-
"""
@file
@brief Implements a quantile non-linear regression.
"""
import inspect
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import DERIVATIVES, LOSS_FUNCTIONS
try:
    from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
except ImportError:  # pragma: no cover
    # scikit-learn < 0.22.
    from sklearn.neural_network.multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.metrics import mean_absolute_error


def absolute_loss(y_true, y_pred):
    """
    Computes the absolute loss for regression.

    :param y_true: array-like or label indicator matrix
        Ground truth (correct) values.
    :param y_pred: array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.
    :return: loss, float
        The degree to which the samples are correctly predicted.
    """
    return np.sum(np.abs(y_true - y_pred)) / y_true.shape[0]


def float_sign(a):
    "Returns 1 if *a > 0*, otherwise -1"
    if a > 1e-8:
        return 1.
    if a < -1e-8:
        return -1.
    return 0.


EXTENDED_LOSS_FUNCTIONS = {'absolute_loss': absolute_loss}
DERIVATIVE_LOSS_FUNCTIONS = {'absolute_loss': np.vectorize(float_sign)}


class CustomizedMultilayerPerceptron(BaseMultilayerPerceptron):
    """
    Customized MLP Perceptron based on
    `BaseMultilayerPerceptron
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py#L40>`_.
    """

    def __init__(self, hidden_layer_sizes, activation, solver,
                 alpha, batch_size, learning_rate, learning_rate_init, power_t,
                 max_iter, loss, shuffle, random_state, tol, verbose,
                 warm_start, momentum, nesterovs_momentum, early_stopping,
                 validation_fraction, beta_1, beta_2, epsilon,
                 n_iter_no_change, max_fun):
        if 'max_fun' in inspect.signature(BaseMultilayerPerceptron.__init__).parameters:
            args = [15000]
        else:
            args = []
        BaseMultilayerPerceptron.__init__(  # pylint: disable=E1121
            self, hidden_layer_sizes, activation, solver, alpha, batch_size,
            learning_rate, learning_rate_init, power_t, max_iter, loss,
            shuffle, random_state, tol, verbose, warm_start, momentum,
            nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2,
            epsilon, n_iter_no_change, *args)

    def _get_loss_function(self, loss_func_name):
        """
        Returns the loss functions.

        @param      loss_func_name      loss function name, see
                                        :epkg:`sklearn:neural_networks:MLPRegressor`
        """
        return LOSS_FUNCTIONS.get(loss_func_name, EXTENDED_LOSS_FUNCTIONS[loss_func_name])

    def _modify_loss_derivatives(self, last_deltas):
        """
        Modifies the loss derivatives.

        @param      last_deltas     last deltas is the difference between the output and the expected output
        @return                     modified derivatives
        """
        if self.loss == 'absolute_loss':
            return DERIVATIVE_LOSS_FUNCTIONS['absolute_loss'](last_deltas)
        return last_deltas  # pragma: no cover

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """
        Computes the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        :param y: array-like, shape (n_samples,)
            The target values.
        :param activations: list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        :param deltas: list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        :param coef_grads: list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        :param intercept_grads: list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        :return: loss, float
        :return: coef_grads, list, length = n_layers - 1
        :return: intercept_grads, list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss_function = self._get_loss_function(loss_func_name)
        loss = loss_function(y, activations[-1])
        # Add L2 regularization term to loss
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # We insert the following modification to modify the gradient
        # due to the modification of the loss function.
        deltas[last] = self._modify_loss_derivatives(deltas[last])

        # Compute gradient for the last layer
        temp = self._compute_loss_grad(  # pylint: disable=E1111
            last, n_samples, activations, deltas, coef_grads, intercept_grads)
        if temp is None:
            # recent version of scikit-learn
            # Compute gradient for the last layer
            self._compute_loss_grad(
                last, n_samples, activations, deltas, coef_grads, intercept_grads)

            inplace_derivative = DERIVATIVES[self.activation]
            # Iterate over the hidden layers
            for i in range(self.n_layers_ - 2, 0, -1):
                deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
                inplace_derivative(activations[i], deltas[i - 1])

                self._compute_loss_grad(
                    i - 1, n_samples, activations, deltas, coef_grads,
                    intercept_grads)
        else:  # pragma: no cover
            coef_grads, intercept_grads = temp  # pylint: disable=E0633

            # Iterate over the hidden layers
            for i in range(self.n_layers_ - 2, 0, -1):
                deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
                inplace_derivative = DERIVATIVES[self.activation]
                inplace_derivative(activations[i], deltas[i - 1])

                coef_grads, intercept_grads = self._compute_loss_grad(  # pylint: disable=E1111,E0633
                    i - 1, n_samples, activations, deltas, coef_grads,
                    intercept_grads)

        return loss, coef_grads, intercept_grads


class QuantileMLPRegressor(CustomizedMultilayerPerceptron, RegressorMixin):
    """
    Quantile MLP Regression or neural networks regression
    trained with norm :epkg:`L1`. This class inherits from
    :epkg:`sklearn:neural_networks:MLPRegressor`.
    This model optimizes the absolute-loss using LBFGS or stochastic gradient
    descent. See @see cl CustomizedMultilayerPerceptron and
    @see fn absolute_loss.

    :param hidden_layer_sizes: tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    :param activation: {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns :math:`f(x) = x`
        - 'logistic', the logistic sigmoid function,
          returns :math:`f(x) = 1 / (1 + exp(-x))`.
        - 'tanh', the hyperbolic tan function,
          returns :math:`f(x) = tanh(x)`.
        - 'relu', the rectified linear unit function,
          returns :math:`f(x) = \\max(0, x)`.
    :param solver: ``{'lbfgs', 'sgd', 'adam'}``, default 'adam'
        The solver for weight optimization.
        - *'lbfgs'* is an optimizer in the family of quasi-Newton methods.
        - *'sgd'* refers to stochastic gradient descent.
        - *'adam'* refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba
        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.
    :param alpha: float, optional, default 0.0001
        :epkg:`L2` penalty (regularization term) parameter.
    :param batch_size: int, optional, default 'auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`
    :param learning_rate: {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning rate schedule for weight updates.
        - 'constant' is a constant learning rate given by
          'learning_rate_init'.
        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)
        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.
        Only used when solver='sgd'.
    :param learning_rate_init: double, optional, default 0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.
    :param power_t: double, optional, default 0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.
    :param max_iter: int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.
    :param shuffle: bool, optional, default True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.
    :param random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param tol: float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.
    :param verbose: bool, optional, default False
        Whether to print progress messages to stdout.
    :param warm_start: bool, optional, default False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    :param momentum: float, default 0.9
        Momentum for gradient descent update.  Should be between 0 and 1. Only
        used when solver='sgd'.
    :param nesterovs_momentum: boolean, default True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.
    :param early_stopping: bool, default False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least ``tol`` for
        ``n_iter_no_change`` consecutive epochs.
        Only effective when solver='sgd' or 'adam'
    :param validation_fraction: float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True
    :param beta_1: float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'
    :param beta_2: float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'
    :param epsilon: float, optional, default 1e-8
        Value for numerical stability in adam. Only used when solver='adam'
    :param n_iter_no_change: int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'

    Fitted attributes:

    * `loss_`: float
        The current loss computed with the loss function.
    * `coefs_`: list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.
    * `intercepts_`: list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.
    * `n_iter_`: int,
        The number of iterations the solver has ran.
    * `n_layers_`: int
        Number of layers.
    * `n_outputs_`: int
        Number of outputs.
    * `out_activation_`: string
        Name of the output activation function.
    """

    def __init__(self,
                 hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10,
                 **kwargs):
        """
        See :epkg:`sklearn:neural_networks:MLPRegressor`
        """
        sup = super(QuantileMLPRegressor, self)  # pylint: disable=R1725
        if "max_fun" not in kwargs:
            sig = inspect.signature(sup.__init__)
            if "max_fun" in sig.parameters:
                kwargs['max_fun'] = 15000
        sup.__init__(hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation, solver=solver, alpha=alpha,
                     batch_size=batch_size, learning_rate=learning_rate,
                     learning_rate_init=learning_rate_init, power_t=power_t,
                     max_iter=max_iter, loss='absolute_loss', shuffle=shuffle,
                     random_state=random_state, tol=tol, verbose=verbose,
                     warm_start=warm_start, momentum=momentum,
                     nesterovs_momentum=nesterovs_momentum,
                     early_stopping=early_stopping,
                     validation_fraction=validation_fraction,
                     beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                     n_iter_no_change=n_iter_no_change, **kwargs)

    def predict(self, X):
        """
        Predicts using the multi-layer perceptron model.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        :return: y : array-like, shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        if hasattr(self, '_predict'):
            y_pred = self._predict(X)
        else:
            y_pred = self._forward_pass_fast(X)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _validate_input(self, X, y, incremental, reset=False):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y

    def score(self, X, y, sample_weight=None):
        """
        Returns mean absolute error regression loss.

        :param X: array-like, shape = (n_samples, n_features)
            Test samples.
        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.
        :param sample_weight: array-like, shape = [n_samples], optional
            Sample weights.
        :return: score, float
            mean absolute error regression loss
        """
        pred = self.predict(X)
        return mean_absolute_error(y, pred, sample_weight=sample_weight)
