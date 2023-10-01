# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
@file
@brief Implements a quantile non-linear regression.
"""
import random
from sklearn.neural_network import MLPRegressor


class LinkedMLPBase:
    """
    Overloads methods from :epkg:`sklearn:neural_networks:MLPRegressor`
    and insert the logic to train linked coefficients.
    """

    def _initialize(self, y, layer_units, dtype):
        super()._initialize(y, layer_units, dtype)
        if hasattr(self, "linked_"):
            return
        if self.linked is None:
            self.linked_ = None
            return
        if isinstance(self.linked, int):

            def _get_random(layer, selected, n_sel):
                indices = []
                c = self.coefs_[layer]
                for i in range(c.shape[0]):
                    for j in range(c.shape[1]):
                        key = layer, "c", i, j
                        if key in selected:
                            continue
                        indices.append(key)
                c = self.intercepts_[layer]
                for i in range(c.shape[0]):
                    key = layer, "i", i
                    if key in selected:
                        continue
                    indices.append(key)

                random.shuffle(indices)
                inds = []
                pos = 0
                nis = set()
                while len(inds) < n_sel and pos < len(indices):
                    ind = indices[pos]
                    if ind[2] in nis:
                        pos += 1
                        continue
                    inds.append(pos)
                    nis.add(ind[2])
                    pos += 1
                return tuple(indices[p] for p in inds)

            n_coefs = sum(
                [c.size for c in self.coefs_] + [c.size for c in self.intercepts_]
            )
            linked = []
            selected = set()
            unchanged = 0
            while len(linked) < n_coefs and unchanged < 10:
                layer = random.randint(0, len(self.coefs_) - 1)
                inds = _get_random(layer, selected, self.linked)
                if len(inds) <= 1:
                    unchanged += 1
                    continue
                unchanged = 0
                for i in inds:
                    selected.add(i)
                linked.append(inds)
            self.linked_ = linked
            self._fix_links(self.coefs_, self.intercepts_)
        elif isinstance(self.linked, list):
            self.linked_ = self.linked
            self._fix_links(self.coefs_, self.intercepts_)
        else:
            raise TypeError(f"Unexpected type for linked {type(self.linked)}.")

    def _fix_links(self, coefs, intercepts):
        if self.linked_ is None:
            return
        for links in self.linked_:
            if len(links) <= 1:
                raise RuntimeError(f"Unexpected value for link {links}.")
            total = 0
            for key in links:
                if key[1] == "c":
                    v = coefs[key[0]][key[2:]]
                else:
                    v = intercepts[key[0]][key[2]]
                total += v
            total /= len(links)
            for key in links:
                if key[1] == "c":
                    coefs[key[0]][key[2:]] = total
                else:
                    intercepts[key[0]][key[2]] = total

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        batch_loss, coef_grads, intercept_grads = super()._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads
        )
        self._fix_links(coef_grads, intercept_grads)
        return batch_loss, coef_grads, intercept_grads


class LinkedMLPRegressor(LinkedMLPBase, MLPRegressor):
    """
    A neural networks regression for which a subset a coefficients
    share the same value. In practice, it should make the training
    more stable. See parameter *linked*.

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
    :param linked: can be a float to defined the ratio of linked coefficients,
        or list of set of indices

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

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="sgd",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        linked=None,
    ):
        """
        See :epkg:`sklearn:neural_networks:MLPRegressor`
        """
        sup = super(LinkedMLPRegressor, self)  # pylint: disable=R1725
        sup.__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.linked = linked
