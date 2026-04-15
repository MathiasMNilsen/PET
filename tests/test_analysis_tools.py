'''
Tests for the analysis tools module.
'''
import pytest
import numpy as np
import pipt.misc_tools.analysis_tools as atools
from misc.structures import PETDataFrame




# ---------------------------------------------------------------------------
# TESTS: remove_outliers
# ---------------------------------------------------------------------------

def _make_pred(arr: np.ndarray) -> PETDataFrame:
    """Wrap a (n_obs, ne) array in a single-cell ensemble PETDataFrame."""
    return PETDataFrame({'y': [arr]}, index=[0], is_ensemble=True)


def _make_obs(arr: np.ndarray) -> PETDataFrame:
    """Wrap a (n_obs,) array in a single-cell observation PETDataFrame."""
    return PETDataFrame({'y': [arr]}, index=[0], is_ensemble=False)


def test_remove_outliers_no_outliers_unchanged():
    """When all members are well-behaved, nothing should be replaced."""
    rng = np.random.default_rng(0)
    ny, ne, nx = 6, 20, 10

    d = rng.standard_normal(ny)
    Y = d[:, None] + rng.standard_normal((ny, ne)) * 0.1  # tight spread
    X = rng.standard_normal((nx, ne))

    Y_df = _make_pred(Y)
    d_df = _make_obs(d)
    pred_out, X_out = atools.remove_outliers(Y_df, d_df, X.copy())

    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(pred_out.at[0, 'y'], Y)


def test_remove_outliers_detects_single_outlier():
    """An injected outlier member should be replaced; good members should be untouched."""
    rng = np.random.default_rng(42)
    ny, ne, nx = 8, 30, 5

    d = rng.standard_normal(ny)
    Y = d[:, None] + rng.standard_normal((ny, ne)) * 0.05  # tight spread
    X = rng.standard_normal((nx, ne))

    # Inject one member far from the observations
    outlier_col = 7
    Y[:, outlier_col] += 1000.0


    Y_df = _make_pred(Y.copy())
    d_df = _make_obs(d)
    np.random.seed(0)
    pred_out, X_out = atools.remove_outliers(Y_df, d_df, X.copy())

    good = np.delete(np.arange(ne), outlier_col)

    # Outlier column in X must have changed
    assert not np.allclose(X_out[:, outlier_col], X[:, outlier_col]), \
        "Outlier state column should have been replaced"

    # Replaced column must equal one of the good source columns
    assert any(np.allclose(X_out[:, outlier_col], X[:, g]) for g in good), \
        "Replaced column should be a copy of a good member"

    # All good members' state columns must be unchanged
    np.testing.assert_array_equal(X_out[:, good], X[:, good])


def test_remove_outliers_pred_replaced_consistently():
    """pred_out cell for the outlier column should match the replacement source."""
    rng = np.random.default_rng(7)
    ny, ne, nx = 5, 20, 4
    
    d  = rng.standard_normal(ny)
    Y = d[:, None] + rng.standard_normal((ny, ne)) * 0.05
    X = rng.standard_normal((nx, ne))

    outlier_col = 3
    Y[:, outlier_col] += 500.0

    np.random.seed(1)
    pred_out, X_out = atools.remove_outliers(
        _make_pred(Y.copy()), _make_obs(d), X.copy()
    )
    good = np.delete(np.arange(ne), outlier_col)

    # Find which good member replaced the state
    src = next(g for g in good if np.allclose(X_out[:, outlier_col], X[:, g]))

    # pred_out's outlier column should match pred's replacement column
    np.testing.assert_array_equal(
        pred_out.at[0, 'y'][:, outlier_col],
        Y[:, src],
    )


def test_remove_outliers_multiple_outliers():
    """Multiple injected outliers should all be replaced."""
    rng = np.random.default_rng(99)
    n_obs, ne, nx = 6, 100, 8  # large ensemble so 3 outliers are a small fraction
    obs  = rng.standard_normal(n_obs)
    pred = obs[:, None] + rng.standard_normal((n_obs, ne)) * 0.05

    outlier_cols = [2, 15, 30]
    for c in outlier_cols:
        pred[:, c] += 1000.0

    X = rng.standard_normal((nx, ne))
    np.random.seed(2)
    pred_out, X_out = atools.remove_outliers(
        _make_pred(pred.copy()), _make_obs(obs), X.copy()
    )

    good = np.setdiff1d(np.arange(ne), outlier_cols)

    for c in outlier_cols:
        assert not np.allclose(X_out[:, c], X[:, c]), \
            f"Outlier column {c} should have been replaced"

    np.testing.assert_array_equal(X_out[:, good], X[:, good])


def test_remove_outliers_explicit_data_var():
    """Passing explicit data_var as a 1-D array should still detect the outlier."""
    rng = np.random.default_rng(5)
    n_obs, ne, nx = 6, 25, 4
    obs  = rng.standard_normal(n_obs)
    pred = obs[:, None] + rng.standard_normal((n_obs, ne)) * 0.05

    outlier_col = 10
    pred[:, outlier_col] += 1000.0

    X = rng.standard_normal((nx, ne))
    # Provide variance that matches the tight spread
    data_var = np.full(n_obs, 0.05**2)

    np.random.seed(3)
    pred_out, X_out = atools.remove_outliers(
        _make_pred(pred.copy()), _make_obs(obs), X.copy(), data_var=data_var
    )

    assert not np.allclose(X_out[:, outlier_col], X[:, outlier_col]), \
        "Outlier should be detected when explicit data_var is supplied"


def test_remove_outliers_output_types_and_shapes():
    """Output types and shapes must match the inputs regardless of outliers."""
    rng = np.random.default_rng(11)
    n_obs, ne, nx = 5, 15, 6
    obs  = rng.standard_normal(n_obs)
    pred = obs[:, None] + rng.standard_normal((n_obs, ne)) * 0.1

    X = rng.standard_normal((nx, ne))

    pred_out, X_out = atools.remove_outliers(
        _make_pred(pred.copy()), _make_obs(obs), X.copy()
    )

    assert isinstance(pred_out, PETDataFrame)
    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == X.shape
    assert pred_out.at[0, 'y'].shape == pred.shape