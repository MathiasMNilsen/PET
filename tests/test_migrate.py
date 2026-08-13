"""Tests for `pet migrate` and the config-schema change it implements."""

import pytest

from pet_cli.__main__ import main
from pet_cli.migrate import MigrationReport, migrate_config, migrate_section

LEGACY_TOML = """\
[dataassim]
daalg = ["esmda", "esmda"]
analysis = "approx"
energy = 0.99

[fwdsim]
parallel = 1
datatype = ["pressure"]
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# ----------------------------------------------------------------------
# Section-level migration
# ----------------------------------------------------------------------

def test_section_daalg_to_scheme():
    report = MigrationReport()
    section = migrate_section({"daalg": ["esmda", "esmda"], "analysis": "approx"}, report)
    assert section["scheme"] == "esmda"
    assert "daalg" not in section
    assert report.changed


def test_section_keeps_second_entry_and_warns_on_mismatch():
    """The second entry is the one that selected the class historically."""
    report = MigrationReport()
    section = migrate_section({"daalg": ["enrml", "lmenrml"], "analysis": "full"}, report)
    assert section["scheme"] == "lmenrml"
    assert any("differing entries" in w for w in report.warnings)


def test_section_accepts_bare_string():
    report = MigrationReport()
    assert migrate_section({"daalg": "esmda", "analysis": "approx"}, report)["scheme"] == "esmda"


def test_section_warns_when_analysis_missing():
    report = MigrationReport()
    migrate_section({"daalg": ["esmda", "esmda"]}, report)
    assert any("analysis" in w for w in report.warnings)


def test_section_without_daalg_is_untouched():
    report = MigrationReport()
    section = migrate_section({"scheme": "esmda", "analysis": "approx"}, report)
    assert section == {"scheme": "esmda", "analysis": "approx"}
    assert not report.changed


def test_section_leaves_unexpected_daalg_alone():
    report = MigrationReport()
    section = migrate_section({"daalg": 42}, report)
    assert section["daalg"] == 42
    assert report.warnings


# ----------------------------------------------------------------------
# File-level migration
# ----------------------------------------------------------------------

def test_migrate_toml_writes_backup(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    report = migrate_config(path)
    assert report.changed
    assert (tmp_path / "case.toml.bak").exists()
    assert "scheme" in path.read_text()
    assert "daalg" not in path.read_text()


def test_migrate_preserves_other_keys(tmp_path):
    import tomli

    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    migrate_config(path)
    with open(path, "rb") as handle:
        cfg = tomli.load(handle)
    assert cfg["dataassim"]["analysis"] == "approx"
    assert cfg["dataassim"]["energy"] == 0.99
    assert cfg["fwdsim"]["datatype"] == ["pressure"]


def test_migrate_dry_run_writes_nothing(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    before = path.read_text()
    report = migrate_config(path, dry_run=True)
    assert report.changed
    assert path.read_text() == before
    assert not (tmp_path / "case.toml.bak").exists()


def test_migrate_is_idempotent(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    migrate_config(path)
    assert not migrate_config(path).changed


def test_migrate_yaml(tmp_path):
    import yaml

    path = _write(
        tmp_path, "case.yaml",
        "dataassim:\n  daalg: [esmda, esmda]\n  analysis: approx\n",
    )
    migrate_config(path)
    cfg = yaml.safe_load(path.read_text())
    assert cfg["dataassim"]["scheme"] == "esmda"


def test_migrate_rejects_unsupported_format(tmp_path):
    path = _write(tmp_path, "case.pipt", "DATAASSIM\n")
    with pytest.raises(ValueError, match="only .toml and .yaml"):
        migrate_config(path)


def test_migrate_handles_optim_section(tmp_path):
    path = _write(tmp_path, "case.toml", '[optim]\ndaalg = ["esmda", "esmda"]\nanalysis = "approx"\n')
    assert migrate_config(path).changed


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_migrate(tmp_path, capsys):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    assert main(["migrate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "scheme = 'esmda'" in out
    assert ".bak" in out


def test_cli_migrate_no_backup(tmp_path):
    path = _write(tmp_path, "case.toml", LEGACY_TOML)
    assert main(["migrate", str(path), "--no-backup"]) == 0
    assert not (tmp_path / "case.toml.bak").exists()


def test_cli_migrate_missing_file(capsys):
    assert main(["migrate", "nope.toml"]) == 1
    assert "no such file" in capsys.readouterr().err


def test_cli_migrate_already_current(tmp_path, capsys):
    path = _write(tmp_path, "case.toml", '[dataassim]\nscheme = "esmda"\nanalysis = "approx"\n')
    assert main(["migrate", str(path)]) == 0
    assert "already on the current schema" in capsys.readouterr().out


# ----------------------------------------------------------------------
# init_da must point users at the tool rather than failing cryptically
# ----------------------------------------------------------------------

def test_init_da_rejects_legacy_daalg_with_migration_hint():
    from pipt import pipt_init

    with pytest.raises(ValueError, match="pet migrate"):
        pipt_init.init_da({"daalg": ["esmda", "esmda"], "analysis": "approx"}, {}, None)


def test_init_da_accepts_new_scheme_key():
    from pipt import pipt_init
    from pipt.update_schemes import registry

    class Spy:
        def __init__(self, da, en, sim):
            self.ok = True

    registry.register_scheme("spy", "approx", Spy)
    try:
        obj = pipt_init.init_da({"scheme": "spy", "analysis": "approx"}, {}, None)
        assert obj.ok
    finally:
        registry.SCHEMES.pop(("spy", "approx"), None)


def test_init_da_rejects_non_string_scheme():
    from pipt import pipt_init

    with pytest.raises(ValueError, match="as a string"):
        pipt_init.init_da({"scheme": ["esmda"], "analysis": "approx"}, {}, None)
