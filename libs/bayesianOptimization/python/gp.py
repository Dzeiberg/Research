""" gp.py

Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp
from pyswarm import pso
from scipy.stats import norm
from scipy.optimize import minimize

def lowerConfidenceBound(x, gaussian_process, n_params=2, return_negative=False):
    # Want to maximize the LCB
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    LCB = 2 * sigma - mu
    if return_negative:
        LCB = -1 * LCB
    return LCB


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1, return_negative=False):
    """
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values of the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    loss_optimum = np.max(evaluated_loss)
    with np.errstate(divide='ignore'):
        gamma = (loss_optimum - mu) / sigma
        expected_improvement = sigma * (gamma * norm.cdf(gamma)) + norm.pdf(gamma)
        expected_improvement[sigma == 0] = 0
    if return_negative:
        expected_improvement *= -1
    return expected_improvement
        

# def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
#     """ expected_improvement

#     Expected improvement acquisition function.

#     Arguments:
#     ----------
#         x: array-like, shape = [n_samples, n_hyperparams]
#             The point for which the expected improvement needs to be computed.
#         gaussian_process: GaussianProcessRegressor object.
#             Gaussian process trained on previously evaluated hyperparameters.
#         evaluated_loss: Numpy array.
#             Numpy array that contains the values of the loss function for the previously
#             evaluated hyperparameters.
#         greater_is_better: Boolean.
#             Boolean flag that indicates whether the loss function is to be maximised or minimised.
#         n_params: int.
#             Dimension of the hyperparameter space.

#     """

#     x_to_predict = x.reshape(-1, n_params)

#     mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

#     if greater_is_better:
#         loss_optimum = np.max(evaluated_loss)
#     else:
#         loss_optimum = np.min(evaluated_loss)

#     scaling_factor = (-1) ** (not greater_is_better)

#     # In case sigma equals zero
#     with np.errstate(divide='ignore'):
#         Z = scaling_factor * (mu - loss_optimum) / sigma
#         expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         expected_improvement[sigma == 0.0] = 0.0

#     return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, bounds, maximizeAQ=True):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acqisition_func: the function that we are optimizing
        gaussian_process: model
        bounds: [n_params, 2] matrix of lb ub pairs for each parameter
        maximizeAQ: Whether the next sample should be the one that maximizes the acquisition function
    """
    #PSO minimizes the acquisition function so we want to minimize the negative LCB (a.k.a. maximize the LCB)
    xopt, fopt = pso(acquisition_func, bounds[:,0], bounds[:,1], args=(gaussian_process, bounds.shape[0], maximizeAQ))
    return xopt

def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=2,
                          gp_params=None, random_search=100, epsilon=1e-7):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.RBF()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            lcb = lowerConfidenceBound(x_random, model, yp, n_params=n_params)
            # ei = expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(lcb), :]
        else:
            next_sample = sample_next_hyperparameter(lowerConfidenceBound, model, bounds, maximizeAQ=True)
        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            print("gp.py: Duplicate in next_sample, choosing next sample randomly")
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp
