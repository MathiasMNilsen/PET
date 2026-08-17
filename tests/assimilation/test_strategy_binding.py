"""Binding an analysis strategy to a scheme instead of mixing it in.

Groundwork for making ``analysis`` a parameter of one scheme class rather than
the thing that selects which of eighteen classes you get. The blocker is that
the strategies read their context -- ``lam``, ``trunc_energy``,
``localization``, ``keys_da``, ``cov_data``, ``scale_data``, ``proj`` -- off
``self``, which only resolves while they are mixed into the scheme. Bound
strategies reach the same context by delegation.

The load-bearing test is
:func:`test_bound_strategy_matches_mixed_in_result`: bound and mixed-in must
produce bit-identical steps, or the collapse would silently change every
scheme's numbers.
"""

import numpy as np
import pytest

from pipt.update_schemes.analysis import AnalysisStrategy
from pipt.update_schemes.analysis.registry import (
    available_strategies,
    get_strategy,
    register_strategy,
)
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
from pipt.update_schemes.update_methods_ns.subspace_update import subspace_update


class FakeLocalization:
    name = None


class FakeScheme:
    """The context an analysis strategy reads, and nothing else.

    Worth recording: the context is wider than the list in
    ``analysis/base.py``. ``full_update`` also reads ``prior_enX``, ``Am``,
    ``ext_Am`` and ``state_scaling`` -- and ``prior_enX`` is *ensemble* state,
    which resolved under the mixin only because the scheme delegates to its
    ensemble. Anything binding strategies has to supply these too.
    """

    def __init__(self, ne=8, nx=5, lam=0.0, trunc_energy=0.99):
        self.lam = lam
        self.trunc_energy = trunc_energy
        self.keys_da = {}
        self.localization = FakeLocalization()
        self.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
        # Context `full_update` needs on top of the documented set.
        self.prior_enX = np.random.default_rng(7).standard_normal((nx, ne))
        self.Am = None
        self.state_scaling = np.ones(nx)


def _case(seed=0, nx=5, ny=4, ne=8):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((nx, ne)),
        rng.standard_normal((ny, ne)),
        rng.standard_normal((ny, ne)),
    )


# ----------------------------------------------------------------------
# Delegation
# ----------------------------------------------------------------------
def test_bound_strategy_reads_context_from_scheme():
    scheme = FakeScheme(lam=3.5, trunc_energy=0.77)
    strategy = approx_update(scheme)

    assert strategy.lam == 3.5
    assert strategy.trunc_energy == 0.77
    assert strategy.localization.name is None


def test_unbound_strategy_resolves_nothing():
    """Optional context must keep falling back to its default.

    The strategies read optional context as ``getattr(self, 'scale_state',
    <default>)``. If an unbound strategy resolved anything, those defaults
    would stop applying.
    """
    strategy = approx_update()

    with pytest.raises(AttributeError):
        strategy.lam
    assert getattr(strategy, "scale_state", "fallback") == "fallback"


def test_binding_does_not_swallow_genuine_attribute_errors():
    scheme = FakeScheme()
    strategy = approx_update(scheme)

    with pytest.raises(AttributeError):
        strategy.no_such_attribute_anywhere


def test_public_writes_go_through_to_the_scheme():
    """Mixed in, every `self.x = ...` in a strategy set it on the scheme.

    Binding has to reproduce that: `subspace_update` delivers its result by
    assigning `w_step`, and the scheme applies it only if `hasattr(self,
    'w_step')`. Without write-through the update is skipped silently.
    """
    scheme = FakeScheme(lam=1.0)
    strategy = approx_update(scheme)

    strategy.lam = 99.0
    assert scheme.lam == 99.0


def test_private_writes_stay_on_the_strategy():
    scheme = FakeScheme()
    strategy = approx_update(scheme)

    strategy._local = "mine"
    assert not hasattr(scheme, "_local")


def test_unbound_writes_stay_local():
    strategy = approx_update()
    strategy.w_step = 5
    assert strategy.w_step == 5


# ----------------------------------------------------------------------
# Equivalence with the mixin path -- the one that matters
# ----------------------------------------------------------------------
@pytest.mark.parametrize("flavour", ["approx", "full", "subspace"])
def test_bound_strategy_matches_mixed_in_result(flavour):
    """Bound and mixed-in must agree bit-for-bit.

    This is what makes collapsing the eighteen classes safe: if the two paths
    diverged, every scheme's numbers would move with no test to catch it.
    """
    strategy_cls = get_strategy(flavour)
    enX, enY, enE = _case()

    # Mixed in: `self` is the scheme, context resolves by inheritance.
    class MixedIn(FakeScheme, strategy_cls):
        pass

    mixed = MixedIn()
    mixed.iteration = 0
    mixed_step = mixed.update(enX=enX, enY=enY, enE=enE)

    # Bound: context resolves by delegation.
    scheme = FakeScheme()
    scheme.iteration = 0
    bound_step = strategy_cls(scheme).update(enX=enX, enY=enY, enE=enE)

    if mixed_step is not None or bound_step is not None:
        np.testing.assert_array_equal(
            np.asarray(bound_step, dtype=float),
            np.asarray(mixed_step, dtype=float),
            err_msg=(
                f"{flavour}: bound and mixed-in return values disagree, so "
                f"collapsing the per-flavour classes would change the numerics."
            ),
        )

    # Side effects are the real payload for some flavours: subspace_update
    # delivers via `w_step` and returns nothing useful, full_update caches `Am`.
    # Comparing only return values would have missed that entirely.
    for attr in ("w_step", "Am"):
        assert hasattr(scheme, attr) == hasattr(mixed, attr), (
            f"{flavour}: bound path {'set' if hasattr(scheme, attr) else 'did not set'} "
            f"{attr} but mixed-in path did the opposite"
        )
        if hasattr(mixed, attr) and getattr(mixed, attr) is not None:
            np.testing.assert_array_equal(
                np.asarray(getattr(scheme, attr), dtype=float),
                np.asarray(getattr(mixed, attr), dtype=float),
                err_msg=f"{flavour}: bound and mixed-in disagree on {attr}",
            )


def test_mixin_path_is_untouched_by_the_new_init():
    """Adding __init__ to AnalysisStrategy must not perturb the mixin MRO.

    Nothing in the scheme's __init__ chain calls super().__init__(), so
    AnalysisStrategy.__init__ is never invoked there and `_scheme` is never
    set -- which is exactly why mixed-in lookup is unaffected.
    """
    class MixedIn(FakeScheme, approx_update):
        pass

    mixed = MixedIn(lam=2.0)
    assert "_scheme" not in vars(mixed)
    assert mixed.lam == 2.0


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def test_registry_resolves_the_shipped_flavours():
    assert get_strategy("approx") is approx_update
    assert get_strategy("subspace") is subspace_update
    assert available_strategies() == ["approx", "full", "subspace"]


def test_registry_is_case_insensitive():
    assert get_strategy("APPROX") is approx_update


def test_unknown_flavour_lists_the_valid_ones():
    with pytest.raises(KeyError, match="Unknown analysis flavour 'nope'"):
        get_strategy("nope")


def test_registering_a_duplicate_needs_overwrite():
    class Extra(AnalysisStrategy):
        def update(self, enX, enY, enE, **kwargs):
            return None

    with pytest.raises(ValueError, match="already registered"):
        register_strategy("approx", Extra)


def test_register_and_resolve_an_out_of_tree_flavour():
    from pipt.update_schemes.analysis import registry

    class Extra(AnalysisStrategy):
        def update(self, enX, enY, enE, **kwargs):
            return None

    register_strategy("extra_flavour", Extra)
    try:
        assert get_strategy("extra_flavour") is Extra
    finally:
        del registry.STRATEGIES["extra_flavour"]
