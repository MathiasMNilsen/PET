# Phase 8 handover: migrating the schemes onto `AssimilationSchemeBase`

Working document for the last phase of the PIPT/POPT convergence refactor.
It exists so a fresh session (or a different person) can pick the work up cold
without re-deriving the context, and without re-discovering the traps listed at
the bottom.

**Status: COMPLETE.** All five steps are done, plus the follow-on that made the
analysis flavour a parameter rather than a class name. Kept as a record of how
the migration was sequenced and, more usefully, of the traps in §7 -- several of
which bit again during the work and are still live hazards for anyone touching
this code.

What actually happened, against the plan below:

- The ordering in §4 was inverted. `es`/`enkf` turned out to be the only schemes
  with no runtime coverage, and `enkf` could not run at all, so `esmda` and
  `enrml` went first -- migrating against the characterisation suite instead of
  against nothing.
- `es` and `enkf` could not be separated: `es_approx` inherits its
  `calc_analysis` from `enkf`, so steps 1 and 2 were one slice.
- Three bugs surfaced that were not part of the refactor: `enkf` reading a
  `full_cov_data` that nothing assigns, ES discarding its own update, and the
  multilevel scheme never having completed a run. See CHANGELOG.md.
- The open question in §8 was answered: `Assimilate` was deleted outright, with
  no deprecated wrapper.

---

## 1. The goal

PIPT schemes currently *inherit* the ensemble:

```python
class esmdaMixIn(Ensemble):        # the scheme IS a data container
    def calc_analysis(self): ...
    def check_convergence(self): ...
```

and an external `pipt.loop.assimilation.Assimilate` object owns the iteration
loop. POPT does the opposite: `OptimizerBase` owns its loop and *composes* with
what it operates on. Phase 8 brings PIPT into line:

```python
class ESMDA(AssimilationSchemeBase):   # the scheme HAS an ensemble
    def update_step(self) -> bool: ...
```

The target base class already exists, is documented, and is covered by 12 unit
tests: `src/pipt/update_schemes/scheme_base.py`.

| `OptimizerBase` (popt, existing) | `AssimilationSchemeBase` (pipt, ready) |
| --- | --- |
| `update_step() -> bool` (abstract) | `update_step() -> bool` (abstract) |
| `run_optimization()` | `run_assimilation()` |
| `check_function_convergence()` | `check_misfit_convergence()` |
| `check_state_convergence()` | `check_state_convergence()` |
| `check_convergence()` (subclass hook) | `check_convergence()` (subclass hook) |
| `Optimizer.minimize(...)` | `Scheme.assimilate(...)` |
| `OptimizeResult` | `AssimilationResult` |

---

## 2. What is already in place

| Piece | Where | Why it matters here |
| --- | --- | --- |
| Characterisation tests | `tests/assimilation/test_numerical_characterisation.py` | Proves a refactor did not change the numbers. **This is what makes Phase 8 safe.** |
| Reference data | `tests/assimilation/characterisation_reference.npz` | Committed golden values for 5 scheme/flavour combinations |
| Target base class | `src/pipt/update_schemes/scheme_base.py` | The contract to migrate onto |
| Ensemble package | `src/pipt/ensembles/` | The collaborator the schemes will compose with |
| Scheme registry | `src/pipt/update_schemes/registry.py` | Dispatch; should need **no** changes during Phase 8 |
| Import-cycle guard | `tests/test_import_hygiene.py` | Phase 8 moves imports around; this catches layering inversions |

---

## 3. Step 0 — close the `forecast()` gap first

`AssimilationSchemeBase` documents an ensemble collaborator protocol requiring
`ensemble.forecast()`. **That method does not exist yet.** Forecasting currently
lives on `Assimilate`:

- `Assimilate.calc_forecast` (line ~340)
- `sim_to_pred_data`, `post_process_forecast`
- `_apply_prediction_scaling`, `_apply_sim2seis_scaling`,
  `_scale_sparse_sim2seis`, `_scale_dense_sim2seis`
- `_apply_sparse_compression`, `_load_restart_prediction_if_available`,
  `_save_forecast_debug`, `_save_reconstructed_forecast_if_requested`

That is roughly 150 lines, and it is ensemble work, not loop work — it uses
`self.ensemble.sim`, `self.ensemble.compress_manager`, `self.ensemble.pred_data`.

**Do this before touching any scheme:** move those methods onto
`AssimilationEnsemble` (or a `ForecastMixin` in `pipt/ensembles/`, matching how
`CompressionMixin` and `LocalAnalysisMixin` were split), exposing a public
`forecast()`. Have `Assimilate.calc_forecast` delegate to it so nothing breaks
yet. Verify with the characterisation suite. Commit separately.

Skipping this and migrating a scheme first will not work — the scheme's
`update_step()` has nowhere to get a forecast from.

---

## 4. Ordering

One scheme per sitting, easiest first. Counts are ensemble-attribute accesses
(`self.enX`, `self.keys_da`, `self.data_df`, …) that each become
`self.ensemble.<attr>`:

| Order | File | Accesses | Lines | Notes |
| --- | --- | --- | --- | --- |
| 0 | `pipt/loop/assimilation.py` | — | ~150 moved | Step 0 above: forecast onto the ensemble |
| 1 | `update_schemes/es.py` | 10 | 103 | Thin layer over `enkf`; do it first to establish the pattern |
| 2 | `update_schemes/enkf.py` | 48 | 198 | |
| 3 | `update_schemes/esmda.py` | 54 | 435 | Also has `log_update`, `_ext_inflation_param`, `_ext_assim_steps` |
| 4 | `update_schemes/enrml.py` | 216 | 1028 | A third of the work. Four scheme classes: `lmenrmlMixIn`, `gnenrmlMixIn`, `co_lm_enrml`, `gn_enrml` |
| 5 | `pipt/loop/assimilation.py` | — | 492 | Retire what is left, once nothing inherits `Ensemble` |

`co_lm_enrml` is deliberately inactive (kept, not star-exported, not in the
registry). Migrate it last or leave it on the old path — do not delete it, the
maintainer asked for it to stay.

`gnenrml_margis` depends on a private `margIS_update` package that is not in
this repo; an inert placeholder stands in. Keep it registered.

---

## 5. Contract translation

Current per-scheme methods, and where they go:

| Today | Target |
| --- | --- |
| `calc_analysis()` | Body moves into `update_step()` |
| `check_convergence() -> (conv, success, why_stop)` | Split: `success` becomes `update_step()`'s return; `conv` becomes `check_convergence() -> bool`; `why_stop` goes into `self.why_stop` |
| `log_update(success, prior_run)` | Keep as-is; call from `update_step()` |
| `self.iteration` bookkeeping | Owned by the base class — remove local increments |

The `success` flag matters: LM schemes **reject** a step and retry with a larger
damping parameter. The base class models this — `update_step()` returning
`False` leaves the iteration counter untouched and retries, with a
`max_rejected` guard so a scheme cannot loop forever refusing its own updates.

---

## 6. Definition of done, per slice

A slice is finished only when all of these hold:

```sh
# 1. numerics unchanged  -- the important one
python -m pytest tests/assimilation/test_numerical_characterisation.py -q

# 2. nothing else regressed
python -m pytest -q                      # expect 267 passed, 1 skipped (or more)

# 3. lint clean (CI runs this)
ruff check src tests

# 4. imports still layered correctly
python -m pytest tests/test_import_hygiene.py -q
```

**Do not regenerate `characterisation_reference.npz` to make a failure go away.**
A behaviour-preserving refactor must produce *no* diff. Regenerate only when a
numerical change is intended, and review the diff before committing.

---

## 7. Traps already hit (do not rediscover these)

1. **The suite is non-deterministic unless seeded.** Schemes perturb
   observations from the *global* `numpy.random` state. Unseeded, repeated runs
   of the same case differ by up to **0.366** in the posterior state. The
   characterisation tests seed `np.random` and force `parallel = 1`. If you add
   cases, do the same.

2. **`parallel > 1` breaks reproducibility.** Keep characterisation cases
   single-threaded.

3. **Not every "unused" variable is unused.** In
   `tests/optimization/test_ensembles.py`, `g0 = ensemble.gradient(...)` looks
   like a dead assignment; the call is load-bearing because `hessian()` reuses
   the ensemble it populates. Deleting it raises `TypeError`. Ruff's autofix
   would have removed it. Check before accepting an autofix in stochastic code.

4. **`ensemble` must not import `pipt`/`popt` at module level.** It is the
   foundation package both build on. A module-level import inverts the layering
   and makes `import ensemble` fail as a first import — that bug lived for a
   long time because the full suite happened to import in a lucky order, and
   only single-file runs exposed it. `tests/test_import_hygiene.py` guards it.

5. **Mechanical splits misplace imports.** When the ensemble was split, three
   imports landed in the wrong module and a `super(Ensemble, self)` call kept
   the old class name. Ruff caught both; tests did not. Run ruff after every
   move.

6. **Class names are public API but changeable.** The maintainer confirmed
   `lmenrml_*` / `esmda_*` may be renamed, but they are imported directly in
   user scripts, so any rename needs a deprecation alias and a note in
   `CHANGELOG.md`.

---

## 8. Open questions for the maintainer

- **SimulatorWraps**: where does it live, and is it pip-installable? The POPT
  tutorial needs `simulator.opm.flow` and the `npv` cost function, both moved
  out of this repo (commit `97b70cd`). Blocks the tutorial fix, not Phase 8.
- Should `Assimilate` be deleted outright at the end, or kept as a thin
  deprecated wrapper for one release?

---

## 9. How to start a session on this

Paste something like:

> Read `docs/phase8_handover.md`. Do Step 0 (move forecast onto the ensemble),
> then stop and show me the diff before touching any scheme.

Then, per slice:

> Read `docs/phase8_handover.md`. Migrate `<file>` onto `AssimilationSchemeBase`
> per the ordering table. The characterisation tests must pass unchanged.
