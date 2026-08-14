"""Compatibility shim for the old ensemble location.

The assimilation ensemble now lives in :mod:`pipt.ensembles`, mirroring how
:mod:`popt.ensembles` is laid out. This module re-exports it so existing
imports keep working::

    from pipt.loop.ensemble import Ensemble        # still fine
    from pipt.ensembles import AssimilationEnsemble  # preferred

Prefer the new path in new code.
"""

from pipt.ensembles import AssimilationEnsemble
from pipt.ensembles import AssimilationEnsemble as Ensemble

__all__ = ["Ensemble", "AssimilationEnsemble"]
