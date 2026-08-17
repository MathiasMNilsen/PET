"""Tests for the shared analysis-strategy base.

The three analysis flavours used to each carry a private copy of ``solve`` and
``sqrtm``. Those copies had drifted: ``approx_update`` used ``A.ndim`` while the
others used ``np.ndim(A)``, so only the latter tolerated a covariance supplied
as a plain list or scalar. These tests pin the consolidated behaviour.
"""

import numpy as np
import pytest

from pipt.update_schemes.analysis import AnalysisStrategy
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
from pipt.update_schemes.update_methods_ns.full_update import full_update
from pipt.update_schemes.update_methods_ns.subspace_update import subspace_update

FLAVOURS = [approx_update, full_update, subspace_update]


@pytest.mark.parametrize("flavour", FLAVOURS, ids=lambda c: c.__name__)
def test_flavours_share_the_strategy_base(flavour):
    assert issubclass(flavour, AnalysisStrategy)


@pytest.mark.parametrize("flavour", FLAVOURS, ids=lambda c: c.__name__)
def test_flavours_no_longer_define_private_helpers(flavour):
    """Helpers must come from the base, not a per-file copy."""
    assert "solve" not in vars(flavour)
    assert "sqrtm" not in vars(flavour)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        AnalysisStrategy()


# ----------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------

def test_solve_diagonal_matches_dense_equivalent():
    diag = np.array([2.0, 4.0])
    B = np.array([[1.0, 3.0], [2.0, 8.0]])
    np.testing.assert_allclose(
        AnalysisStrategy.solve(diag, B),
        AnalysisStrategy.solve(np.diag(diag), B),
    )


def test_solve_dense_is_a_true_inverse_apply():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    B = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(A @ AnalysisStrategy.solve(A, B), B, atol=1e-12)


def test_solve_accepts_list_covariance():
    """Regression: approx_update's old `A.ndim` raised AttributeError here."""
    out = AnalysisStrategy.solve([2.0, 4.0], np.ones((2, 2)))
    np.testing.assert_allclose(out, [[0.5, 0.5], [0.25, 0.25]])


# ----------------------------------------------------------------------
# sqrtm
# ----------------------------------------------------------------------

def test_sqrtm_diagonal():
    np.testing.assert_allclose(AnalysisStrategy.sqrtm(np.array([4.0, 9.0])), [2.0, 3.0])


def test_sqrtm_accepts_list():
    np.testing.assert_allclose(AnalysisStrategy.sqrtm([4.0, 9.0]), [2.0, 3.0])


def test_sqrtm_dense_squares_back():
    A = np.array([[4.0, 0.0], [0.0, 9.0]])
    root = AnalysisStrategy.sqrtm(A)
    np.testing.assert_allclose(root @ root, A, atol=1e-10)


# ----------------------------------------------------------------------
# The mixin products must keep working unchanged
# ----------------------------------------------------------------------

def test_historical_names_still_select_their_flavour():
    """The per-flavour names keep their meaning, by holding rather than being.

    BREAKING: these classes used to *inherit* their strategy, so
    ``issubclass(esmda_approx, approx_update)`` held. Collapsing the eighteen
    classes into five made the flavour a parameter, so the alias now binds an
    approx_update *instance*. What matters -- which strategy it uses -- is
    unchanged, and that is what this asserts.
    """
    from pipt.update_schemes import esmda_approx, gnenrml_subspace, lmenrml_full
    from pipt.update_schemes.analysis.registry import get_strategy

    for scheme, flavour_name, flavour_cls in [
        (esmda_approx, "approx", approx_update),
        (lmenrml_full, "full", full_update),
        (gnenrml_subspace, "subspace", subspace_update),
    ]:
        assert scheme.FLAVOUR == flavour_name
        assert get_strategy(scheme.FLAVOUR) is flavour_cls
        assert not issubclass(scheme, AnalysisStrategy), (
            f"{scheme.__name__} should hold a strategy, not inherit one"
        )
