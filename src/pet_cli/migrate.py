"""Migrate legacy PET config files to the current schema.

Currently handles one change: the two-element ``daalg`` key, which packed an
assimilation family and an update method into a list, is replaced by a single
``scheme`` key naming the algorithm::

    daalg = ["esmda", "esmda"]   ->   scheme = "esmda"

The second element was the one that actually selected the class, so that is
what carries over. Where the two elements disagree the second still wins, and
the migration reports it so the change is visible rather than silent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import tomli
import tomli_w
import yaml

__all__ = ["migrate_config", "migrate_section", "MigrationReport"]

_DA_SECTIONS = ("dataassim", "optim")


class MigrationReport:
    """What a migration changed, or would change."""

    def __init__(self) -> None:
        self.changes: list[str] = []
        self.warnings: list[str] = []

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def __str__(self) -> str:
        lines = [f"  - {c}" for c in self.changes]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def migrate_section(section: dict, report: MigrationReport) -> dict:
    """Migrate one config section in place, recording what changed."""
    if "daalg" not in section:
        return section

    daalg = section.pop("daalg")

    if isinstance(daalg, str):
        scheme = daalg
    elif isinstance(daalg, (list, tuple)) and daalg:
        scheme = daalg[-1]
        if len(daalg) == 2 and daalg[0] != daalg[1]:
            report.warnings.append(
                f"daalg was {list(daalg)!r} with differing entries; "
                f"kept {scheme!r}, which is the one that selected the class."
            )
        elif len(daalg) > 2:
            report.warnings.append(
                f"daalg had {len(daalg)} entries {list(daalg)!r}; kept {scheme!r}."
            )
    else:
        report.warnings.append(
            f"daalg had unexpected value {daalg!r}; left the file unchanged."
        )
        section["daalg"] = daalg
        return section

    section["scheme"] = scheme
    report.changes.append(f"daalg = {daalg!r}  ->  scheme = {scheme!r}")

    if "analysis" not in section:
        report.warnings.append(
            "No 'analysis' key found; add one to select the analysis flavour "
            "(e.g. analysis = 'approx')."
        )

    return section


def _load(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with open(path, "rb") as handle:
            return tomli.load(handle), "toml"
    if suffix in (".yaml", ".yml"):
        with open(path) as handle:
            return yaml.safe_load(handle), "yaml"
    raise ValueError(
        f"Cannot migrate '{path}': only .toml and .yaml/.yml are supported. "
        f"Convert legacy .pipt/.popt files first with `pet convert`."
    )


def _dump(config: dict, path: Path, fmt: str) -> None:
    if fmt == "toml":
        with open(path, "wb") as handle:
            tomli_w.dump(config, handle)
    else:
        with open(path, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)


def migrate_config(path, *, dry_run: bool = False, backup: bool = True) -> MigrationReport:
    """Migrate a config file to the current schema.

    Parameters
    ----------
    path : str or Path
        Path to a ``.toml`` or ``.yaml`` config file.
    dry_run : bool, optional
        Report what would change without writing anything.
    backup : bool, optional
        Keep the original alongside the migrated file as ``<name>.bak``.

    Returns
    -------
    MigrationReport
    """
    path = Path(path)
    config, fmt = _load(path)
    report = MigrationReport()

    if not isinstance(config, dict):
        raise ValueError(f"'{path}' does not contain a mapping at the top level.")

    for name in _DA_SECTIONS:
        section = config.get(name)
        if isinstance(section, dict):
            migrate_section(section, report)

    if report.changed and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        _dump(config, path, fmt)

    return report
