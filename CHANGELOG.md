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

- **`pipt.loop.assimilation.Assimilate` is removed, with no shim.** Schemes own
  their iteration loop now, as popt's optimizers do. The whole `pipt.loop`
  package is gone, including the `pipt.loop.ensemble` compatibility shim.

  ```python
  # before                                  # after
  from pipt.loop.assimilation import Assimilate
  scheme = pipt_init.init_da(kd, ke, sim)   scheme = ESMDA(kd, ke, sim)
  Assimilate(scheme).run()                  result = scheme.assimilation_loop()
  ```

  `pipt_init.init_da(...)` still works and still returns the scheme; only the
  driver changed. `Scheme.assimilate(kd, ke, sim)` is the one-line form.

- **Eighteen scheme classes collapsed into five.** `ESMDA`, `EnKF`, `ES`,
  `LMEnRML` and `GNEnRML` are classes taking `analysis` as an argument, and
  replace the factory functions of the same names. The per-flavour names remain
  importable as thin subclasses pinning their flavour.

  One consequence is not source-compatible: those classes used to *inherit*
  their strategy, so `issubclass(esmda_approx, approx_update)` held. They now
  *hold* one, so it is `False`. Behaviour and numbers are unchanged; only the
  type relationship goes. A class cannot both be one of five and be-a
  per-flavour strategy.

- **The config's `analysis` key is no longer overridden by a default.**
  `build_scheme`/`ESMDA(...)` took `analysis="approx"` as a parameter default
  and never consulted the config, so a config asking for `subspace` silently
  built the `approx` scheme through that entry point while `init_da` built the
  right one. Precedence is now explicit argument, then config, then `"approx"`.

- **Analysis strategies moved** from `pipt.update_schemes.update_methods_ns` to
  `pipt.update_schemes.analysis`, joining the base class and registry that
  already lived there. Modules are renamed to `approx`/`full`/`subspace`/
  `hybrid`/`margis`; the class names are unchanged.

  This affects code outside this repository: `enrml.py` walked
  `update_methods_ns` with `pkgutil` so a private namespace package could supply
  `margIS_update` alongside the shipped placeholder. A private overlay must now
  target `pipt.update_schemes.analysis`, or the inert placeholder is used
  instead — silently.

- **`iterinfo` hooks receive the scheme**, not the removed `Assimilate` object.
  Custom `main(self)` hooks reading loop attributes need adjusting.

- **Scheme machinery moved to `pipt.update_schemes.core`** —
  `AssimilationSchemeBase`, `StrategyMixin`, `AssimilationWorkflowMixin` — so
  `pipt.update_schemes` lists algorithms rather than mixing them with the
  scaffolding they stand on.

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

- **`AssimilationSchemeBase`** (`pipt.update_schemes.core`) — the PIPT
  counterpart to popt's `OptimizerBase`, with a matching contract
  (`update_step`/`assimilation_loop`/`check_*_convergence`/`assimilate`). The
  ensemble is a collaborator rather than a superclass. Every scheme is now
  migrated onto it.

- **`AnalysisStrategy`** (`pipt.update_schemes.analysis`) — shared base for the
  approx/full/subspace flavours, the counterpart to popt's `subroutines`.

- **`pipt.localization`** — replaces the 888-line `cov_regularization` monolith
  with a package: an ABC and config builder, one module per strategy, and a
  factory dispatching on a `name` attribute.

- **`pet` command line**: `validate`, `convert`, `migrate`, `version`.

- **`ensemble.checkpoint.RestartMixin`** — checkpoint/restart logic shared by
  PIPT and POPT rather than duplicated.

### Fixed

- **`analysisdebug` could not record the prior.** Every scheme computed its
  prior misfit inside the first `calc_analysis`, which runs *after* the
  iteration-0 artifacts are written. So `debug_analysis_step_0.npz` never
  contained `ensemble_misfit`, `data_misfit` or `prior_data_misfit`; the run
  printed `Cannot save ensemble_misfit, because it is a local variable!` and
  carried on. Prior scoring moved to a new `score_prior()` hook that the loop
  calls between the prior forecast and `after_prior_forecast`, so step 0 is
  described by the same attributes as every later step. Numbers are unchanged
  — the characterisation suite pins all nine scheme/flavour combinations.

  Two consequences beyond the saved files:

  - LM-EnRML and GN-EnRML no longer recompute `prior_data_misfit` from the
    *rejected* forecast each time they reject their first step. The old
    `iteration == 0` branch also re-clobbered `data_misfit` right after
    `score_and_commit` had restored it.
  - `ensemble_misfit` is now set by EnKF, ES and the multilevel hybrid too;
    only ES-MDA and the EnRML pair kept it before.

- **`save_folder` in a `dataassim` block was silently ignored.** Only the
  unspaced `savefolder` was read, so a config using the underscored spelling —
  which popt's optimizers accept — wrote to the default `Results` folder
  instead. Both spellings are now accepted.
- **ES discarded its own update.** The posterior came back bit-identical to the
  prior: the analysis ran, the forecast ran, the log reported a reduced misfit,
  but the state promotion sat inside an equal-misfit branch that is essentially
  never taken, so `enX_temp` was never committed. Anyone running ES was handed
  their prior ensemble back.
- **`enkf` could not run at all.** `check_convergence` read
  `self.full_cov_data`, which nothing assigns, so every run raised
  `AttributeError` at the end of its first iteration. Commit 6401e6e rewrote the
  two sibling call sites to use `scale_data` and missed this one.
- **The multilevel scheme had never completed a run.** Four faults: the level
  loop iterated ensemble *sizes* while using the value as an *index*;
  `treat_modeling_error` was called before `pred_data` existed;
  `calc_analysis` overwrote the step `hybrid_update` had just computed with the
  `None` it returns, discarding every update; and `esmda_hybrid` relied on C3
  linearisation to reach the scheme's `__init__`, which stopped happening when
  schemes left the ensemble hierarchy. It now runs end to end.
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

- **Local analysis is broken along both routes.** `localization = {name =
  "localanalysis"}` reaches a branch that warns and returns `None`, so no update
  is applied and the run completes reporting a misfit — the posterior is the
  prior. Separately, `LocalAnalysisMixin` calls `self._ext_obs()`, which is
  defined nowhere in the codebase.
- **`es`/`enkf` with `analysis="subspace"`** raise `ValueError: Length of values
  (11) does not match length of index (15)`. `esmda/subspace` is unaffected, so
  the fault is in the sequential path.
- **The GIES schemes cannot be constructed.** `GIESMixIn.__init__` uses the
  pre-ensemble-matrix API (`self.state`, `self.obs_data`) and calls
  `self._ext_obs()`, which does not exist. Reproduced unchanged before the
  Phase 8 work, so this predates it.
- `docs/tutorials/pipt/tutorial_pipt.ipynb` has been updated to the current API
  but **not re-executed** — running it needs the OPM `flow` simulator, so its
  stored outputs are from the old code.
- `docs/tutorials/popt/tutorial_popt.ipynb` imports `popt.loop.optimize`,
  `popt.update_schemes.enopt` and `popt.cost_functions.npv`, none of which
  exist — popt now provides `optimization_methods/` and `ensembles/`, and the
  NPV cost function moved to the simulator wrappers. Pre-existing; the
  published POPT tutorial cannot run. Fixing it needs the notebook re-executed
  against the OPM `flow` simulator.
