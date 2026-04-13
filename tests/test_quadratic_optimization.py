from popt.loop.ensemble_gaussian import GaussianEnsemble
from popt.update_schemes.enopt import EnOpt
from popt.update_schemes.linesearch import LineSearch
from popt.cost_functions.quadratic import quadratic
from scipy.optimize import rosen

import numpy as np
import os


dim = 2
kwens = {
    'ne': 10,
    'transform': True,
    'natural_gradient': False,
    'controls': {
        'x': {'mean': [5]*dim, 'var': 1.0e-5, 'limits': [-10, 10]}
    }
}

kwopt = {
    'maxiter': 50,
    'tol': 1e-2,
    'alpha': 0.25,
    'alpha_maxiter': 4,
    'resample': 0,
    'optimizer': 'GD',
    'restartsave': False,
    'restart': False,
    'save_data': ['alpha', 'obj_func_values']
}


def test_quadratic_enopt(temp_examples_dir):
    np.random.seed(101122)
    os.chdir(temp_examples_dir)
    
    ensemble = GaussianEnsemble(kwens, None, quadratic)
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()
    enopt = EnOpt(
        ensemble.function, 
        x0, 
        args=(cov,), 
        jac=ensemble.gradient, 
        hess=ensemble.hessian, 
        bounds=bounds, 
        **kwopt
    )
    state = ensemble.get_state()
    obj = enopt.obj_func_values
    np.testing.assert_array_almost_equal(state, [0.5, 0.5], decimal=1)
    np.testing.assert_array_almost_equal(obj, [0.0], decimal=1)


def test_quadratic_linesearch(temp_examples_dir):
    np.random.seed(101122)
    os.chdir(temp_examples_dir)
    
    # Create ensemble
    ensemble = GaussianEnsemble(kwens, None, quadratic)

    # Get initial state
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    
    # Run Optimization
    res = LineSearch( 
        x=x0,
        fun=ensemble.function,
        jac=ensemble.gradient,
        args=(cov,), 
        bounds=bounds, 
    )

    np.testing.assert_array_almost_equal(res.x, [0.5, 0.5], decimal=1)
    np.testing.assert_almost_equal(res.fun, 0.0, decimal=4)


def test_rosenbrock_linesearch(temp_examples_dir):
    np.random.seed(10_08_1997)
    os.chdir(temp_examples_dir)

    dim = 100
    kw = {
        'ne': 100,
        'transform': False,
        'natural_gradient': False,
        'controls': {
            'x': {'mean': [-2]*dim, 'var': 0.001, 'limits': [-2, 2]}
        }
    }

    # Define objective function
    func = lambda x, *args, **kwargs: rosen(x)

    # Create ensemble
    ensemble = GaussianEnsemble(kw, None, func)

    # Get initial state
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    options = {
        'maxiter': 1000,
        'step_size': 1.0,
        'ftol': 1e-8,
    }

    # Run Optimization
    res = LineSearch( 
        x=x0,
        fun=ensemble.function,
        jac=ensemble.gradient,
        args=(cov,), 
        bounds=bounds,
        method='BFGS',
        **options
    )
    
    np.testing.assert_array_almost_equal(res.x, np.ones(dim), decimal=0)
    assert np.linalg.norm(res.x - np.ones(dim)) < 0.1*np.sqrt(dim)

