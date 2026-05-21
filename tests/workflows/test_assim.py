"""
Tests for Data Assimilation workflows using the Van der Pol oscillator as a test case.
"""
import os
import yaml
import pytest
import numpy as np
import pandas as pd

from simulator.vanderpol import VanDerPolOscillator, _integrate
from pipt.loop.assimilation import Assimilate
from input_output import read_config
from pipt import pipt_init

@pytest.fixture
def num_cores():
    '''
    Returns the number of CPU cores to use for parallel runs in tests.
    Uses half of the available cores, but at least 1.
    '''
    n = max(os.cpu_count()//2, 1)
    return n


def _setup(seed=12345):
    rng = np.random.default_rng(seed)

    # True state
    x1_0, x2_0, mu = 1.0, 0.0, 1.0

    # Make prior ensemble
    ne = 1000
    X1 = 0.05 + 0.1 * rng.standard_normal(ne)
    X2 = 0.05 + 0.1 * rng.standard_normal(ne)
    MU = 1.5  + 0.5 * rng.standard_normal(ne)
    np.savez("prior_ensemble.npz", 
        x1=X1[np.newaxis,:], 
        x2=X2[np.newaxis,:], 
        mu=MU[np.newaxis,:]
    )

    # Observation times and report points
    time_steps = np.arange(0, 16, 1, dtype=float)       # 0..15
    report_points = np.arange(1, 16, 1, dtype=int)      # 1..15

    # True run
    res = _integrate(x1_0, x2_0, mu, time_steps, atol=1e-5, rtol=1e-5)

    # Perturb observations with noise, (observations are x1 at report points)
    sigma = 0.1
    obs = res[report_points, 0] + sigma * rng.standard_normal(len(report_points))

    # DataFrame for true observations
    df_true = pd.DataFrame({"x1": obs}, index=report_points)
    df_true.index.name = "steps"
    df_true.to_pickle("true_data.pkl")

    # Variance DataFrame in PET format    
    variance = sigma ** 2
    df_var = pd.DataFrame(
        {"x1": [f"['abs', {variance}]" for _ in range(len(report_points))]},
        index=report_points,
    )
    df_var.index.name = "steps"
    df_var.to_pickle("var.pkl")


def _make_config_file(name, kwda, parallel_runs=1):
    kwens = {
        'ne': 1000,
        'state': ['x1', 'x2', 'mu'],
        'importstate': 'prior_ensemble.npz',
        'prior_x1': {'var': 1.0},
        'prior_x2': {'var': 1.0},
        'prior_mu': {'var': 1.0},
    }
    kwsim = {
        'reporttype': 'steps',
        'reportpoints': list(range(1, 16)),
        'datatype': ['x1'],
        'parallel': parallel_runs,
        'compute_adjoints': False,
    }
    config = {
        'ensemble': kwens,
        'dataassim': kwda,
        'fwdsim': kwsim,
    }
    with open(f"{name}.yaml", 'w') as f:
        yaml.dump(config, f)

def _data_mismatch(d, Y, cov):
    n = Y.shape[1]
    dm = 0.0
    for i in range(n):
        r = Y[:, i] - d
        dm += np.squeeze(r.T @ np.linalg.solve(cov, r) / n)
    return dm



def test_EMSDA_approx(tmp_path, num_cores):
    np.random.seed(12345)

    # Make test folder and change to it
    path = tmp_path / "esmda_test"
    path.mkdir()
    os.chdir(path)

    # Setup data and prior ensemble
    _setup(seed=12345)

    # Make config file for EMSDA
    kwda = {
        'daalg': ['esmda', 'esmda'],
        'analysis': 'approx',
        'mda': {'tot_assim_steps': 8, 'inflation_param': 8*[8]},
        'energy': 0.99,
        'obsname': 'steps',
        'data': 'true_data.pkl',
        'datavar': 'var.pkl',
        'save_folder': 'results',
        'analysisdebug': ['state', 'pred_data', 'ensemble_misfit'],
    }
    _make_config_file(name="config_emsda", kwda=kwda, parallel_runs=num_cores)

    # Run assimilation
    cfg_da, cfg_sim, cfg_ens = read_config.read("config_emsda.yaml")
    ensemble = pipt_init.init_da(
        cfg_da, 
        cfg_ens, 
        VanDerPolOscillator(cfg_sim),
    )
    Assimilate(ensemble).run()

    # Check data mismatch
    dm = _data_mismatch(
        d=ensemble.vecObs,
        Y=ensemble.pred_data.to_matrix(),
        cov=np.diag(ensemble.cov_data),
    )
    assert dm < 60.0, f"Data mismatch too high: {dm} >= 60.0"

    # Check mu-parameter
    mu_true = 1.0
    mu_post_mean = ensemble.enX[2, :].mean()
    mu_prior_mean = ensemble.prior_enX[2, :].mean()
    assert abs(mu_post_mean - mu_true) < 0.2*abs(mu_prior_mean - mu_true)


def test_LM_EnRML_approx(tmp_path, num_cores):
    np.random.seed(12345)

    # Make test folder and change to it
    path = tmp_path / "lm_enrml_test"
    path.mkdir()
    os.chdir(path)

    # Setup data and prior ensemble
    _setup(seed=12345)

    # Make config file for LM-EnRML
    kwda = {
        'daalg': ['enrml', 'lmenrml'],
        'analysis': 'approx',
        'iteration': {'max_iter': 8, 'lambda': 10, 'lambda_factor': 5, 'trunc_energy': 0.99},
        'energy': 0.99,
        'obsname': 'steps',
        'data': 'true_data.pkl',
        'datavar': 'var.pkl',
        'save_folder': 'results',
        'analysisdebug': ['state', 'pred_data', 'ensemble_misfit'],
    }
    _make_config_file(name="config_lm_enrml", kwda=kwda, parallel_runs=num_cores)

    # Run assimilation
    cfg_da, cfg_sim, cfg_ens = read_config.read("config_lm_enrml.yaml")
    ensemble = pipt_init.init_da(
        cfg_da, 
        cfg_ens, 
        VanDerPolOscillator(cfg_sim),
    )
    Assimilate(ensemble).run()

    # Check data mismatch
    dm = _data_mismatch(
        d=ensemble.vecObs,
        Y=ensemble.pred_data.to_matrix(),
        cov=np.diag(ensemble.cov_data),
    )
    assert dm < 60.0, f"Data mismatch too high: {dm} >= 60.0"

    # Check mu-parameter
    mu_true = 1.0
    mu_post_mean = ensemble.enX[2, :].mean()
    mu_prior_mean = ensemble.prior_enX[2, :].mean()
    assert abs(mu_post_mean - mu_true) < 0.2*abs(mu_prior_mean - mu_true)


def test_GN_EnRML_approx(tmp_path, num_cores):
    np.random.seed(12345)

    # Make test folder and change to it
    path = tmp_path / "gn_enrml_test"
    path.mkdir()
    os.chdir(path)

    # Setup data and prior ensemble
    _setup(seed=12345)

    # Make config file for GN-EnRML
    kwda = {
        'daalg': ['enrml', 'gnenrml'],
        'analysis': 'approx',
        'iteration': {'max_iter': 8, 'gamma': 0.5, 'gamma_factor': 5, 'trunc_energy': 0.99},
        'energy': 0.99,
        'obsname': 'steps',
        'data': 'true_data.pkl',
        'datavar': 'var.pkl',
        'save_folder': 'results',
        'analysisdebug': ['state', 'pred_data', 'ensemble_misfit'],
    }
    _make_config_file(name="config_gn_enrml", kwda=kwda, parallel_runs=num_cores)

    # Run assimilation
    cfg_da, cfg_sim, cfg_ens = read_config.read("config_gn_enrml.yaml")
    ensemble = pipt_init.init_da(
        cfg_da, 
        cfg_ens, 
        VanDerPolOscillator(cfg_sim),
    )
    Assimilate(ensemble).run()

    # Check data mismatch
    dm = _data_mismatch(
        d=ensemble.vecObs,
        Y=ensemble.pred_data.to_matrix(),
        cov=np.diag(ensemble.cov_data),
    )
    assert dm < 60.0, f"Data mismatch too high: {dm} >= 60.0"

    # Check mu-parameter
    mu_true = 1.0
    mu_post_mean = ensemble.enX[2, :].mean()
    mu_prior_mean = ensemble.prior_enX[2, :].mean()
    dx0 = abs(mu_prior_mean - mu_true)
    dx1 = abs(mu_post_mean - mu_true)
    assert dx1 < 0.2*dx0, f"Parameter improvement too low: {dx1} >= 0.2*{dx0}"
    
