# Execution handover

## One-command campaign

Run `./run_all_baselines.sh` from the repository root. The launcher validates all
three configurations before execution, uses sequential child processes, tees
logs into `outputs/all-baselines/logs/`, and automatically resumes recorded work.
It rejects source/config drift; use `--output outputs/new-campaign` for a new run.
Completed suites are checked before being skipped. Failed/incomplete campaigns
are marked as such and do not publish a combined results gallery.

`./run_all_baselines.sh --dry-run` prepares a read-only plan.
`./run_all_baselines.sh --analyze-only` regenerates saved artifacts without
training. The final `outputs/all-baselines/index.html` links synthetic, frozen
and adaptive results, with combined CSV/JSON and separate-protocol statistics.
Full runs execute the configured training budgets; preparation did not run them.

## What is ready

The synthetic causal suite and a four-environment native JaxMARL suite have
configuration validation, explicit execution commands, budget accounting,
checkpoint saving/loading, frozen and retraining protocols, independent
Monte Carlo evaluation, atomic result records, seed statistics, saved-data
plots, PNG/PDF exports, GIFs and matched CORA/no-op comparisons.

The default application command performs preflight only. This handover did not
execute baseline experiments, learner updates or oracle rollouts.

```bash
.venv/bin/python scripts/prepare_experiments.py
.venv/bin/python -m cora.benchmarks.release
```

`outputs/native-preflight.json` records the current native jobs and commands;
`outputs/finalization-preflight.json` is a historical plan.
Training-step figures there are conservative upper bounds; first-round policies
are shared across organizational methods. Resource use depends on training
length and JAX compilation. Configurations are execution templates, not final
preregistered research protocols or evidence of converged policies.

## Run later

Synthetic causal experiments (Simple Spread remains an explicit legacy path):

```bash
.venv/bin/python -m cora.benchmarks.runner --config configs/bench_research.yaml
```

World Comm, SMAX, OvercookedV2 and JaxNav with frozen continuation policies:

```bash
.venv/bin/python -m cora.benchmarks.release --config configs/applications.yaml --run
```

Adaptive organizational sequences:

```bash
.venv/bin/python -m cora.benchmarks.release --config configs/applications_adaptive.yaml --run
```

The adaptive protocol explicitly reinitializes and retrains a policy between
organizational decisions, freezing it during intervention estimation. Each
method follows its own organizational trajectory with matched training/query
budgets. This is not uninterrupted online optimizer-state continuation. Effect
estimates do not carry across rounds./random policies remain fixed.

Use `--resume` with identical config/source to resume an interrupted application
run. Per-method completion records are written after raw oracle and trace
artifacts. Every oracle reference is evaluator-only and method-specific because
later-round organization/policy contexts differ across methods.

## Analyze without running experiments

```bash
.venv/bin/python -m cora.benchmarks.release --config configs/applications.yaml --analyze-only
.venv/bin/python -m cora.benchmarks.runner --config configs/bench_research.yaml --analyze-only
```

Analysis reads existing raw artifacts. It does not train, simulate or query an
oracle. Figures and replays are generated only when real saved data exists.

## Scientific scope that is not complete

The two unspecified published-comparator reproductions remain unavailable.
`compositional_modular` is an explicitly documented CCOR adaptation of an
additive modular architecture; `organization_mdp` is a generic empirical
one-decision planner. Neither is mislabeled as a faithful paper reproduction.
The one-decision planner reduces to tabular greedy selection in the frozen CCOR
problem; it is not independent evidence against sequential organizational
planning. `planning.py` also provides a general discounted finite-MDP solver.

The sequential suite now uses native JaxMARL simulators, with assignment hints
added to local observations and native goal delegation in JaxNav. World Comm
controls the pursuer team against fixed foragers; team rewards are not
mixed. SMAX controls allies against the upstream heuristic enemy. OvercookedV2
has two organizational duties and only one swap, so cannot demonstrate compound
interaction gains. See native_benchmarks.md for exact action mappings and limits.
Custom application simulators remain historical fixtures only.

Statistical, structural and policy-shift guarantees remain conditional. The
application suite uses diagnostic confidence. Learned-weight cross-domain
transport, distributed consensus and pathwise safety are not claimed.

Thus the supported experimental suite can be launched without further glue
code, but full publication readiness requires comparator reproduction decisions,
scientific protocol review and actual experiments. See specification_audit.md.
