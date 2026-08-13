"""
Command-line interface for PET (Python Ensemble Toolbox).

This CLI does not run simulations itself -- forward simulators and cost
functions are user-supplied Python code and must be wired up in a driver
script (see the PIPT/POPT tutorials). Instead, it covers the parts of a
PET workflow that are purely about configuration files:

    pet validate CONFIG          check a config file for missing/invalid keys
    pet convert CONFIG --to FMT  convert a legacy .pipt/.popt file to toml/yaml
    pet version                  print the installed PET version
"""
from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from input_output import read_config


def _cmd_version(_args: argparse.Namespace) -> int:
    try:
        print(pkg_version("PET"))
    except PackageNotFoundError:
        print("PET (version unknown - not installed as a package)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config_file = args.config_file
    if not Path(config_file).is_file():
        print(f"error: no such file: {config_file}", file=sys.stderr)
        return 1

    try:
        sections = read_config.read(config_file)
    except Exception as err:  # noqa: BLE001 - report any parse failure to the user
        print(f"error: failed to parse '{config_file}': {err}", file=sys.stderr)
        return 1

    names = ["dataassim/optim", "fwdsim", "ensemble"]
    print(f"Parsed '{config_file}' successfully:")
    for name, section in zip(names, sections):
        count = len(section) if section else 0
        print(f"  [{name}] {count} keyword(s)")

    problems = _check_mandatory_keywords(sections)
    if problems:
        print("\nProblems found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nNo problems found.")
    return 0


def _check_mandatory_keywords(sections) -> list[str]:
    """Run the mandatory-keyword checks and collect any failures as messages."""
    cfg_prb = sections[0] or {}
    cfg_sim = sections[1] if len(sections) > 1 else {}
    cfg_ens = sections[2] if len(sections) > 2 else None

    problems: list[str] = []
    checks = [(read_config.check_mand_keywords_fwd, cfg_sim)]
    if "daalg" in cfg_prb:
        checks.append((read_config.check_mand_keywords_da, cfg_prb))
    elif cfg_prb:
        checks.append((read_config.check_mand_keywords_opt, cfg_prb))
    if cfg_ens:
        checks.append((read_config.check_mand_keywords_en, cfg_ens))

    for check, section in checks:
        try:
            check(section)
        except AssertionError as err:
            problems.append(str(err))
    return problems


def _cmd_convert(args: argparse.Namespace) -> int:
    config_file = args.config_file
    if not Path(config_file).is_file():
        print(f"error: no such file: {config_file}", file=sys.stderr)
        return 1

    try:
        if args.to == "toml":
            read_config.convert_txt_to_toml(config_file)
        else:
            read_config.convert_txt_to_yaml(config_file)
    except Exception as err:  # noqa: BLE001 - report any conversion failure to the user
        print(f"error: failed to convert '{config_file}': {err}", file=sys.stderr)
        return 1

    new_file = read_config.change_file_extension(config_file, args.to)
    print(f"Wrote '{new_file}'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pet", description=__doc__.strip().splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="check a config file for missing/invalid keys")
    validate.add_argument("config_file", help="path to a .toml, .yaml, .pipt, or .popt config file")
    validate.set_defaults(func=_cmd_validate)

    convert = subparsers.add_parser("convert", help="convert a legacy .pipt/.popt config file")
    convert.add_argument("config_file", help="path to a .pipt or .popt config file")
    convert.add_argument("--to", choices=["toml", "yaml"], default="toml", help="output format (default: toml)")
    convert.set_defaults(func=_cmd_convert)

    version = subparsers.add_parser("version", help="print the installed PET version")
    version.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
