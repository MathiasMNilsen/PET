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

  Existing `.pipt`/`.popt` files are unaffected until converted.

- **`pipt.loop.assimilation.Assimilate` is removed, with no shim.** Schemes own
  their iteration loop now, as popt's optimizers do. The whole `pipt.loop`
  package is gone, including the `pipt.loop.ensemble` compatibility shim.

  ```python
  # before                                  # after
  from pipt.loop.assimilation import Assimilate
  scheme = pipt_init.init_da(kd, ke, sim)   scheme = ESMDA(kd, ke, sim)
  Assimilate(scheme).run()                  result = scheme.run_assimilation()
  ```

  `pipt_init.init_da(...)` still works and still returns the scheme; only the
  driver changed. `Scheme.assimilate(kd, ke, sim)` is the one-line form.

- **`optimization_loop()` and `assimilation_loop()` are renamed** to
  `run_optimization()` and `run_assimilation()`, with no aliases. `_loop` named
  the mechanism rather than the job — nobody calls it because they want a loop
  — and `assimilation_loop` sat awkwardly beside the `run_forecast` /
  `run_prior_forecast` already on the same class. The rename affects both
  packages so they keep the same shape.

  ```python
  # before                       # after
  enopt.optimization_loop()      enopt.run_optimization()
  esmda.assimilation_loop()      esmda.run_assimilation()
  ```

  The class-level shortcuts are unchanged: `EnOpt.minimize(...)` and
  `ESMDA.assimilate(...)` still construct and run in one call.

- **Per-iteration result files renamed.** `debug_analysis_step_{i}.npz` is now
  `assimilation_result_{i}.npz`, the assimilation counterpart of popt's
  `optimize_result_{i}.npz`. The files were never a debugging aid — they are
  the record of a run, one per iteration, with iteration 0 the prior — and the
  old name said otherwise. **Post-processing that globs
  `debug_analysis_step_*` must be updated**; nothing can alias a filename.

  The config key that selects them follows: `analysisdebug` is now `savedata`,
  again matching popt. The old spelling still works and warns, and
  `pet migrate` rewrites it in place alongside `daalg`. There is no `saveit`
  switch to go with it: listing variables turns saving on and omitting the key
  turns it off, so a config cannot name variables that are silently discarded.

  ```toml
  # before                                   # after
  [dataassim]                                [dataassim]
  analysisdebug = ["state", "pred_data"]     savedata = ["state", "pred_data"]
  ```

  `analysis_tools.save_analysisdebug` is likewise deprecated in favour of
  `save_assimilation_result`; the alias writes the new filename, not the old
  one.

- **Eighteen scheme classes collapsed into five, and the per-flavour names
  removed.** `ESMDA`, `EnKF`, `ES`, `LMEnRML` and `GNEnRML` are classes taking
  `analysis` as an argument, and replace both the factory functions of the
  same names and the per-flavour classes (`esmda_approx`, `lmenrml_full`,
  ...): each was one line pinning a flavour the constructor argument already
  expresses. Use `ESMDA(..., analysis="approx")` and friends instead --
  `registry.get_scheme(scheme, analysis)` still resolves a `(scheme,
  analysis)` pair for config-driven code, now to the algorithm class with
  `analysis` pre-bound rather than to a stored class per combination.

  Two combinations are not aliases and keep their own classes: `esmda_hybrid`
  (multilevel ES-MDA) and `gnenrml_margis` (a private, externally-implemented
  strategy) are algorithms in their own right that happen to share a name,
  reachable via `registry.get_scheme("esmda", "hybrid")` /
  `("gnenrml", "margis")`. `esmda_geo` is gone outright: its `__init__` took
  the wrong arguments and referenced an attribute the class never set, so it
  could not have been constructed successfully; nothing exercised it.

  Not source-compatible: the removed classes used to *inherit* their
  strategy, so `issubclass(esmda_approx, approx_update)` held. The replacement
  *holds* one instead. Behaviour and numbers are unchanged -- pinned by the
  characterisation suite -- only the type relationship goes.

  Each algorithm class now declares, right on the class, which flavours it
  supports and which class handles each -- `ESMDA.COMPATIBLE_ANALYSES = {
  "approx": approx_update, "full": full_update, "subspace": subspace_update}`
  -- so reading one scheme's source shows everything it supports, with no
  registry lookup needed to find out. `EnKF`/`ES` requesting `analysis="full"`
  used to resolve to the `approx` strategy only through the per-flavour
  classes; requesting it directly on `EnKF`/`ES` ran the (numerically
  identical, more expensive) `full` strategy. `EnKF.COMPATIBLE_ANALYSES`
  now points `"full"` at the same class as `"approx"`, which is what the
  removed classes' docstrings already claimed ("EnKF/ES take a single step,
  so full and approx coincide") but did not, in fact, apply to direct
  construction. `ES` inherits the dict unchanged, so the fact lives in one
  place and applies regardless of entry point.

  `register_strategy` (`pipt.update_schemes.analysis.registry`) no longer
  makes a newly registered flavour automatically selectable on an existing
  scheme -- each scheme's `COMPATIBLE_ANALYSES` is what a config's `analysis`
  key is actually checked against. Add the flavour to a scheme's dict
  directly, or register a whole `(scheme, analysis)` combination via
  `pipt.update_schemes.registry.register_scheme`.

  `esmda_hybrid` (multilevel ES-MDA) moved off the mixed-in path onto this
  same bound-strategy pattern: `hybrid_update` now inherits `AnalysisStrategy`
  and `esmda_hybrid.COMPATIBLE_ANALYSES = {"hybrid": hybrid_update}`, in place
  of `class esmda_hybrid(hybrid_update, ESMDA)`. Its calling convention
  (`update(enX, enY, enE, **kwargs)`) already matched the bound shape; only
  the values are lists of per-level matrices rather than single ones, which
  the attribute-forwarding that binding relies on does not care about. One
  consequence: `esmda_hybrid.COMPATIBLE_ANALYSES` deliberately does *not*
  include `approx`/`full`/`subspace` -- those strategies expect a single
  `enX`/`proj` matrix, which this scheme's per-level state never gives them;
  requesting one now raises a clear error instead of the previous, unrelated
  behaviour of silently running the hybrid update regardless of what
  `analysis` was asked for. Verified bit-for-bit unchanged against the
  pre-conversion code (no committed reference existed to pin, so this was
  checked directly rather than through the characterisation suite).
  `gnenrml_margis` remains the one scheme still wired up the old way -- see
  below.

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
  `margIS_update` alongside what shipped here. A private overlay must now
  target `pipt.update_schemes.analysis`, or the module below is used instead —
  silently.

  `analysis/margis.py` itself is no longer an inert placeholder: it now
  carries a real port of the margIS math from an older layout, with attribute
  names (`self.ne`, `self.proj`, `self.lam`, `self.scale_data`) matching this
  codebase's current conventions, plus fixes against Stordal, Lorentzen &
  Fossum (2023), *Marginalized iterative ensemble smoothers for data
  assimilation*:

  - **`GNEnRML.calc_analysis` was missing a branch.** `margIS_update`
    delivers its result via `self.W_step` (capital W) -- the ensemble
    *matrix* update ("following e.g. Raanes et al. 2019" in the code this
    was ported from), reconstructed as
    `enX = mean(prior_enX) + prior_enX @ proj * sqrt(ne-1) @ W`. Only the
    lowercase `w_step` *vector* update ("following e.g. Evensen et al. 2019",
    a different reconstruction for a differently-initialised `W`) had
    survived in this codebase's `GNEnRML.calc_analysis`. The first attempt
    at a fix renamed `self.W_step` to `self.w_step` to match what existed --
    which was wrong, and confirmed wrong by running it: routed through the
    vector-update branch, the assimilation made the misfit *worse* by five
    orders of magnitude, unchanged however small the step length shrank --
    the signature of the wrong formula entirely, not a scale problem. Fixed
    properly by restoring the missing `hasattr(self, 'W_step')` branch to
    `GNEnRML.calc_analysis`, gamma-scaled to match the existing `w_step`
    branch's convention, and reverting this file to deliver `self.W_step` as
    it always did.
  - **The first-call check used the wrong iteration convention.** `if
    self.iteration == 1` guarded initialising `current_W`/`current_w`/`D`.
    This codebase's schemes count from `self.iteration = 0` (confirmed
    against `GNEnRML.__init__` and against `subspace_update`, which checks
    `if self.iteration == 0` for the same reason), so initialisation never
    ran and the first real call failed outright with `AttributeError:
    'AssimilationEnsemble' object has no attribute 'current_W'`. Fixed to
    check `== 0`.
  - **The update loop was hardcoded to 70 individual data points**, each its
    own "type" of one (`M = 1`), instead of the paper's Eq. 8/9 sum over
    actual data types with each type's real count as `M`. Now groups rows by
    data type (`self.data_df`'s columns) instead.
  - **It carried its own `scale()`**, duplicating `AnalysisStrategy.solve` --
    the same duplication `approx`/`full`/`subspace` had before they were
    consolidated onto the shared base. Now inherits `AnalysisStrategy` and
    calls `self.solve` directly, picking up the same robustness fix
    consolidation made (`np.ndim` instead of `scaling.shape`, so a covariance
    passed as a plain list or scalar works).

  That inheritance change surfaced a fifth, pre-existing bug, unrelated to
  any of the above: the old `gnenrml_margis(GNEnRML, margIS_update)` listed
  `GNEnRML` first, so `StrategyMixin.update` -- reachable through `GNEnRML`'s
  own MRO chain -- was what plain attribute lookup actually found, not
  `margIS_update.update`, regardless of what `bind_strategy` decided about
  `self.strategy`. That `update` raises immediately for a mixed-in flavour,
  so the scheme could not run at all, independent of anything above.

  **`gnenrml_margis` is gone.** Once `margIS_update` took the same
  `(enX, enY, enE, **kwargs)` shape as every other strategy, mixing it into a
  separate class was no longer the only way to wire it up -- and, as the bug
  above shows, was actively worse than the alternative. `GNEnRML.
  COMPATIBLE_ANALYSES` now has a `"margis"` entry like `"approx"` and friends;
  `GNEnRML(..., analysis="margis")` binds `margIS_update` by ordinary
  composition, the same way `ESMDA(..., analysis="approx")` binds
  `approx_update`, with no MRO shadowing possible because binding never
  touches the class hierarchy. `("gnenrml", "margis")` resolves through the
  generic `ALGORITHMS` + `COMPATIBLE_ANALYSES` path now, not
  `SPECIAL_SCHEMES` -- unlike `("esmda", "hybrid")`, which stays special
  because `esmda_hybrid` really is a distinct class (multilevel ES-MDA), not
  an alias for an existing one.

  Run against real data for the first time (PIPT's own `TinyBox` tutorial
  case, 9 data types across 6 wells): misfit prior 1.96e10, after one
  iteration 1.18e8, a 99.4% reduction -- a large, sensible improvement, not
  just an absence of errors. Still not a golden reference, though: one run,
  one case, no committed values pinning today's numbers the way
  `test_numerical_characterisation` does for the other flavours -- see
  `pipt.update_schemes.analysis.margis`.

- **`iterinfo` hooks receive the scheme**, not the removed `Assimilate` object.
  Custom `main(self)` hooks reading loop attributes need adjusting.

- **Scheme machinery moved to `pipt.update_schemes.core`** —
  `AssimilationSchemeBase`, `AnalysisBindingMixin`, `AssimilationWorkflowMixin`
  — so `pipt.update_schemes` lists algorithms rather than mixing them with the
  scaffolding they stand on.

- **One name for the analysis concept.** The code called the same thing an
  "analysis" (the config key, `COMPATIBLE_ANALYSES`) and a "strategy" (the
  base class, the registry, the bound attribute). It is now "analysis"
  throughout:

  | before | after |
  | --- | --- |
  | `AnalysisStrategy` | `AnalysisBase` |
  | `StrategyMixin` | `AnalysisBindingMixin` |
  | `pipt.update_schemes.core.strategy` | `pipt.update_schemes.core.analysis_binding` |
  | `STRATEGIES` | `ANALYSES` |
  | `get_strategy` / `register_strategy` / `available_strategies` | `get_analysis` / `register_analysis` / `available_analyses` |
  | `bind_strategy()` | `bind_analysis()` |
  | `scheme.strategy` (object) + `scheme.analysis` (name) | `scheme.analysis` (object) + `scheme.analysis_name` (name) |

  Note the last row: `analysis` is both the constructor argument (a flavour
  *name*) and the attribute holding the resulting object, the way
  `Model(optimizer="adam").optimizer` is an optimizer instance.
  `pipt.localization` keeps its own, unrelated use of "strategy".

- **Schemes inherit one base, `AssimilationScheme`,** instead of listing
  `(AssimilationWorkflowMixin, StrategyMixin, AssimilationSchemeBase)`. The
  order was load-bearing and easy to get wrong: the workflow mixin *overrides*
  five hooks (`after_analysis`, `after_forecast`, `after_loop`,
  `after_accepted_iteration`, `after_prior_forecast`) that the base defines as
  no-op defaults, so listing it after the base would have silently stopped
  every run from saving its artifacts. Combining them once removes that
  hazard. `AssimilationWorkflowMixin` stays a usable standalone mixin, and a
  scheme wanting the loop without the artifacts can still subclass
  `AssimilationSchemeBase` directly.

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
  (`update_step`/`run_assimilation`/`check_*_convergence`/`assimilate`). The
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

- A converged `LMEnRML`/`GNEnRML` run reported `no stopping reason recorded`.
  Both schemes set their converged flag in `score_and_commit()` but never set
  `conv_msg`, and they disable the base class's generic criteria -- which are
  the only other thing that sets it. `result.message` and the closing log line
  now name the criterion that fired (the data-misfit tolerance, or
  `lambda_max` for LM-EnRML).

- `LMEnRML`/`GNEnRML` re-armed a convergence criterion they had just
  disabled. Both pass `step_tol=0.0` to switch off the base class's generic
  state-change check, then set `self.step_tol` from config (default `0.01`) a
  few lines later. Neither reads the value itself — the only consumer is the
  check they opted out of. The assignment was vestigial, carried over from the
  never-constructed `co_lm_enrml`/`gn_enrml`, and is removed. No behaviour
  change today, because `check_state_convergence()` cannot fire at all (see
  Known issues).

- `hybrid_update` carried its own `scale()`, a duplicate of the inherited
  `AnalysisBase.solve()` with the arguments in the opposite order. Removed in
  favour of `solve`, which additionally accepts a covariance given as a plain
  list or scalar.

- **`savedata` could not record the prior.** Every scheme computed its
  prior misfit inside the first `calc_analysis`, which runs *after* the
  iteration-0 artifacts are written. So the step-0 file never
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

- **A scheme reaches its ensemble through declared properties, not
  `__getattr__`.** Reads a scheme does not own (`enX`, `pred_data`,
  `keys_da`, `localization`, ...) were forwarded to the ensemble by a blanket
  `__getattr__`, which resolved *any* name, was invisible to `dir()`,
  autocompletion and type checkers, and silently absorbed typos. Each of the
  25 names that actually crosses that boundary is now an explicit `property`
  on `AssimilationSchemeBase`: 21 read-only, plus `cov_data`, `scale_data`,
  `proj` and `Am`, which a scheme may legitimately compute for itself and so
  have setters. Reading is unchanged (`self.enX` still works everywhere);
  *assigning* a read-only one now raises `AttributeError` instead of quietly
  creating a shadow the forecast would never see. Ensemble state is still
  written explicitly through `self.ensemble.<name> = ...`.

- `logit` and `logger_name` are real `[dataassim]` options. Both were
  documented on the scheme base but could never take effect: the ensemble
  built its logger unconditionally, hardcoded to `assim.log`, and every scheme
  overwrote the scheme-side logger with the ensemble's. The ensemble now
  honours both, defaulting to `ASSIM.log`, and `logit = false` installs a
  no-op logger so no file is created at all.

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

- `AssimilationSchemeBase.check_state_convergence()` is inert: `enX_old` is
  initialised to `None` and never assigned, so it returns `False` for every
  scheme. Finishing it means snapshotting `ensemble.enX` before each analysis
  and giving the schemes a `step_tol` they opt into. Documented in place
  rather than deleted, since the criterion itself is wanted.

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
