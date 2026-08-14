"""Guards against import cycles between the top-level packages.

``ensemble`` is the foundation package that both ``pipt`` and ``popt`` build on.
If it imports from either of them at module level, the layering inverts and
importing ``ensemble`` first raises a partially-initialized-module error.

That regression existed for a long time without being noticed, because the full
test suite happened to import the packages in an order that avoided it -- only
running a single test file surfaced it. These tests each import in a fresh
subprocess so import order cannot mask the problem.
"""

import subprocess
import sys

import pytest

TOP_LEVEL_PACKAGES = ["ensemble", "misc", "input_output", "pipt", "popt", "simulator"]


@pytest.mark.parametrize("package", TOP_LEVEL_PACKAGES)
def test_package_imports_standalone(package):
    """Each package must import cleanly as the very first import."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {package}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`import {package}` failed as a first import:\n{result.stderr}"
    )


def test_ensemble_does_not_import_pipt_or_popt_at_module_level():
    """The foundation package must not depend upward at import time.

    Uses a fresh interpreter and checks which modules are resolved: importing
    ``ensemble`` must not drag in ``pipt`` or ``popt``.
    """
    code = (
        "import sys; import ensemble; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('pipt', 'popt'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert not leaked, (
        "Importing `ensemble` pulled in upward dependencies: "
        f"{leaked}. Keep pipt/popt imports inside the functions that use them."
    )
