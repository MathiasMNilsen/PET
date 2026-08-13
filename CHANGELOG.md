# Changelog

All notable changes to PET are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes

- **Config: `daalg` is replaced by `scheme`.** The analysis flavour is a
  parameter of an algorithm rather than a separate algorithm, so the
  two-element `daalg` key has nothing left to encode. Only its second entry
  ever selected the class; the first was a module hint the registry no longer
  needs.

  ```toml
  # before                          # after
  [dataassim]                       [dataassim]
  daalg = ["esmda", "esmda"]        scheme = "esmda"
  analysis = "approx"               analysis = "approx"
  ```

  Loading a config that still uses `daalg` raises an error showing the rewrite
  and naming the migration command. To migrate:

  ```sh
  pet migrate my_config.toml          # rewrites in place, keeps <config>.bak
  pet migrate my_config.toml --dry-run
  pet convert my_case.pipt && pet migrate my_case.toml   # legacy text configs
  ```

  Existing `.pipt`/`.popt` files are unaffected until converted, and the
  concrete scheme classes (`esmda_approx`, `lmenrml_full`, ...) remain
  importable under their existing names.

### Added

- **One constructor per algorithm**, with the flavour as an argument, so five
  names reach what previously took eighteen:

  ```python
  from pipt import ESMDA, available_schemes
  scheme = ESMDA(cfg_da, cfg_en, sim, analysis="approx")
  available_schemes()   # every valid (scheme, analysis) pair
  ```

- **`pipt.update_schemes.registry`** — an explicit scheme table replacing
  dispatch by string surgery. Unknown keys now report the valid alternatives
  instead of failing on a missing attribute. Third-party and private schemes
  can join via `register_scheme()`.

- **`AssimilationSchemeBase`** (`pipt.update_schemes.scheme_base`) — the PIPT
  counterpart to popt's `OptimizerBase`, with a matching contract
  (`update_step`/`assimilation_loop`/`check_*_convergence`/`assimilate`). The
  ensemble is a collaborator rather than a superclass. No scheme is migrated
  onto it yet.

- **`AnalysisStrategy`** (`pipt.update_schemes.analysis`) — shared base for the
  approx/full/subspace flavours, the counterpart to popt's `subroutines`.

- **`pipt.localization`** — replaces the 888-line `cov_regularization` monolith
  with a package: an ABC and config builder, one module per strategy, and a
  factory dispatching on a `name` attribute.

- **`pet` command line**: `validate`, `convert`, `migrate`, `version`.

- **`ensemble.checkpoint.RestartMixin`** — checkpoint/restart logic shared by
  PIPT and POPT rather than duplicated.

### Fixed

- `gies/rlmmac_update.py` imported `_calc_loc` from the removed
  `cov_regularization` module, so importing the GIES-RLMMAC scheme raised
  `ImportError`.
- `co_lm_enrml.calc_analysis` added the imported *function* `aug_state` to an
  ndarray — there is no local variable of that name — raising `TypeError` on
  every run.
- `approx_update.solve` used `A.ndim` where the other two flavours used
  `np.ndim(A)`, so a covariance supplied as a list or scalar raised
  `AttributeError` with that flavour only.
- `convert_txt_to_yaml` opened its output in binary mode while `yaml.dump`
  writes `str`, so every call raised `TypeError`.
- Two uses of `np.bool`, removed in modern NumPy.
- popt's line-search `zoom()` read `aold`/`phi_old` before binding them on the
  first branch.

### Changed

- Packaging: corrected the license path (pointed at a nonexistent
  `LICENSE.txt`), moved test tooling to a `dev` extra, added classifiers and a
  supported-Python floor matching CI.
- CI runs a lint job, previously a `# TODO: Lint` comment.
- Removed 71 unused imports; replaced 33 bare `except:` clauses so
  `KeyboardInterrupt`/`SystemExit` are no longer swallowed. `ruff check src` is
  clean and enforced.
- The legacy `.pipt`/`.popt` parser's nested try/except cascade was rewritten as
  named helpers with identical behaviour.

### Known issues

- `docs/tutorials/popt/tutorial_popt.ipynb` imports `popt.loop.optimize`,
  `popt.update_schemes.enopt` and `popt.cost_functions.npv`, none of which
  exist — popt now provides `optimization_methods/` and `ensembles/`, and the
  NPV cost function moved to the simulator wrappers. Pre-existing; the
  published POPT tutorial cannot run. Fixing it needs the notebook re-executed
  against the OPM `flow` simulator.
