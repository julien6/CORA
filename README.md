# CORA — native JaxMARL experiments

The current baseline and sequential experiment configurations use **World Comm,
SMAX, OvercookedV2 and JaxNav**, backed by the pinned, unmodified JaxMARL
simulators. Warehouse, rescue, wildfire and hospital prototypes are retained as
legacy fixtures/demonstrations and are no longer research experiment defaults.
The synthetic causal suite remains useful for ground-truth validation.

## Run the complete configured campaign

From the repository root, one command starts all configured baseline suites:

```bash
./run_all_baselines.sh
```

This runs the synthetic, native frozen-policy and native adaptive suites
sequentially, then checks result coverage and produces
`outputs/all-baselines/index.html`, combined CSV/JSON, per-suite statistics,
learner checkpoints, raw experiment logs, native screenshots and GIFs. The
configured budgets include real training and can take substantial time. Nothing
is trained by preparation or dry-run. Rerun the same command after an interruption
to resume with unchanged code/configs. Separate suites retain separate estimands.

```bash
./run_all_baselines.sh --dry-run       # no runs and no output creation
./run_all_baselines.sh --analyze-only  # rebuild saved results without training
./run_all_baselines.sh --output outputs/new-campaign  # independent campaign
```

The command covers the runnable configured baselines and documented adaptations.
Unspecified/unavailable external reproductions are not silently substituted.

```bash
bash scripts/setup.sh
# Validate only: no training, oracle queries or experiment rollouts.
.venv/bin/python -m cora.benchmarks.release
.venv/bin/python scripts/prepare_experiments.py
# Run later, when explicitly desired.
.venv/bin/python -m cora.benchmarks.release --config configs/applications.yaml --run
.venv/bin/python -m cora.benchmarks.release --config configs/applications_adaptive.yaml --run
# Rebuild native GIFs, PNG/PDF figures and statistics from real saved data.
.venv/bin/python -m cora.benchmarks.release --config configs/applications.yaml --analyze-only
```

See [docs/native_benchmarks.md](docs/native_benchmarks.md) for exact adapters,
organizational interventions, objectives and limitations, and
[docs/runbook.md](docs/runbook.md) for execution/resume instructions. Frozen and
retraining configurations have **not** been run. The technical check uses
single transitions, trace round-trips, native rendering and abstract trainer
tracing (`jax.eval_shape`); it does not execute optimizer updates.

IPPO, MAPPO, IQL, VDN and QMIX reuse the pinned upstream network/trainer code
through a common discrete, flattened observation adapter. This is an adapted
benchmark interface, not an assertion that all upstream environment-specific
benchmark protocols are reproduced. `imageio` supports the native kitchen
renderer. The principal dependencies and lock file include it.

`configs/native_smoke.yaml` is an optional tiny, random-policy/no-op operational
run; it performs no learning, but still executes real reference rollouts when
`--run` is supplied. No scores or scientific gains are manufactured. The two
unspecified published comparator reproductions remain unresolved; diagnostic
confidence and responsibility hints are not certified safety guarantees.

## Legacy artifacts and entry points

`outputs/bench-smoke/`, `configs/bench_smoke.yaml`, the explicit
`cora.benchmarks.runner` MPE path and the PoC below preserve historical Simple
Spread results and checks. `configs/bench_research.yaml` now selects synthetic
scenarios only. The default `cora-bench` command performs the native suite's
read-only preflight. Existing demonstration galleries remain illustrations,
separate from learned baseline experiments.

## Original minimal PoC

This is an engineering integration smoke test, not an experimental validation of
CORA. It runs a symbolic organization, controlled intervention branches, a
factorized ridge value estimator, diagnostic selection, and one existing IPPO
update on JaxMARL Simple Spread. It asserts no causal identification, safe
improvement, sample efficiency, performance advantage, or convergence.

From the repository root, with Python 3.11 or 3.12 and Git installed:

```bash
bash scripts/setup.sh
.venv/bin/python -m cora.poc
.venv/bin/pytest -q
```

The setup script downloads the separate upstream checkout at
`third_party/JaxMARL`, pins commit
`976aeb152cb184a5095021968bba94da96eb6394`, and installs the tested CPU stack
in `.venv`. `requirements-lock.txt` pins the resolved transitive environment;
`pyproject.toml` pins the principal libraries. Python 3.12.3 was tested here.
Upstream IPPO is loaded from the checkout, which must remain at that revision
with its trainer source unchanged. JaxMARL is Apache-2.0 licensed; its source
and license remain in the separate checkout. No upstream source is modified.
The repository is not currently initialized as a Git repository.

Default output is `outputs/poc/`:

- `dataset.json`: eight intervention outcomes and continuation metadata.
- `summary.json`: configuration, dependencies, estimator/decision diagnostics,
  matched continuation returns, and IPPO metrics.
- `traces.npz`: raw matched continuation positions and rewards.
- `poc.png`: trajectories, assignment overlays, and realized reward traces.

The delivered run passed 23 automated tests. Its summary, dataset, raw traces,
and PNG were identical to a second full same-seed execution. The saved
`outputs/poc/verification.json` and `outputs/tests.log` record this verification.

For the first pass, excluding the learner update:

```bash
.venv/bin/python -m cora.poc --skip-ippo --output outputs/pass1
```

Regenerate the figure from the saved data without rerunning simulation:

```bash
.venv/bin/python -m cora.poc --render-only
```

Use `--seed 8`, `--output ...`, or `--config ...` to override the defaults.
Configuration uses OmegaConf from the existing Hydra ecosystem, without a new
configuration framework. Defaults are CPU, horizon 20, eight data branches,
and ten IPPO rollout steps. Guards prohibit more than 20 steps, more than eight
data branches, and more than one update per execution. The test suite also
executes one update; it does not run long training or performance campaigns.

## Semantics and limits

The authoritative research source currently available is
`work_in_progress_document.pdf` (11 October 2026), especially Sections 3–8 and
Appendices A–B. `paper/paper.tex` is an unmodified Springer sample, not the CORA
manuscript. The supplied minimal PoC roadmap authorizes this smaller scope.

Three agents each have the executor role and exactly one of three landmark
missions. Every mission must have exactly one executor, and role permissions
must allow the assignment. Edits swap agents 0/1, 0/2, and 1/2. Binary compound
edits execute atomically in that declared index order. Overlapping swaps do not
commute; the order is part of the transaction definition, not an inferred causal
property. Different edit sets can produce the same final assignment but have
different intervention costs. All admissible edit sets, including no-op, remain
candidates. The no-op preserves the organization and version exactly.

The wrapper appends the agent's own mission one-hot to the existing 18-value
observation, giving 21 values. All physical transitions, discrete actions, and
native rewards are unchanged. The fixed guidance policy consumes only this
observation, selecting movement toward the assigned visible landmark with 10%
uniform random exploration. No action mask is needed for this minimal language.
This is an explicit organization-conditioned continuation protocol, not a new
environment or a causal discovery method. IPPO consumes the same observations.
Organization persists over the environment's automatic episode resets.

The estimand is discounted finite-horizon **mean native agent reward**,
conditional on one logged reset seed/context and the fixed guidance policy
version. Initial uniform experiments are a randomized permutation without
replacement. The final branch uses deterministic maximum-leverage acquisition
with conditional probability one. All branches use identical simulator reset
state and common rollout keys. These matched outcomes are correlated and the
single-context dataset cannot establish population identification. Context/root
seed, branch keys, policy version, discount, horizon, edit set, resulting
organization, costs, and conditional assignment probabilities are saved.

The fixed hyperedge contains all three edits because physical interaction and
collective rewards may couple them. Its downward-closed monomial family has
eight features, including the intercept and three-way term. This small dense
model makes no sparsity or efficiency claim. Ridge fits raw rollout values;
predicted effect differences subtract the no-op prediction. Costs are deducted
only at selection. The model reports rank and row-space value identifiability;
ridge does not make unsupported values identifiable.

Diagnostic radii are a configurable scale times regularized feature leverage.
They are **not calibrated confidence intervals**. Selection compares the best
pessimistic net prediction with the optimistic no-op, requires supported target
values, and otherwise abstains. Structural and policy-shift bounds remain
unverified. Even an accepted edit is not certified safe. Deterministic active
acquisition supplies no positivity guarantee and does not implement the draft's
full decision-difference acquisition criterion.

IPPO reuses the upstream `baselines/IPPO/ippo_ff_mpe.py` trainer, actor-critic,
GAE, clipped loss, and Optax update. The adapter injects the wrapper and replaces
the privately loaded module's logging reference with a local callback. It does
not initialize WandB or send metrics remotely. One environment, one epoch, one
minibatch, and one update are enforced. The adapter checks finite parameters,
optimizer state and metrics, changed parameters, and finite nonzero applied
gradients recovered from Adam's first moment. Upstream's MPE logging wrapper
scales episode-return logging by agent count; CORA estimation uses native mean
reward. The upstream metric is not a CORA value estimate.

Decentralized action execution is exercised; organizational commit is centralized.
The IPPO update is a separate integration smoke test, not a continuation-policy
update interleaved with causal data collection. No estimated effects are carried
across a policy update. Privileged positions are accessed only for offline
visualization; learning/selection receive features and logged scalar returns.

Modules under `src/cora/` separate organization, interventions, environment
adapter, features, estimation, exploration, selection, rollouts, upstream learner
adapter, and output generation. Tests exercise permission failures, transaction
ordering, native environment semantics, reproducibility, exact small-factorial
reconstruction, omitted-interaction failure, coefficient versus value
identifiability, acceptance/abstention, dataset integrity, and the full pipeline.
Confidence coverage and theoretical guarantees are deferred because the PoC does
not implement valid error bounds. See `technical_report.txt` for the audit and
actual verification record.

## Extended low-resource validation

The specification ledger is [docs/specification_audit.md](docs/specification_audit.md).
Known temporal-SCM validation, conditional confidence formulas, policy-version
isolation, a sampled 64-edit backend, and four small scripted application
prototypes are available. They extend technical coverage, not scientific claims.

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/validate_extended.py
```

The second command produces `outputs/extended-smoke/index.html`, raw decisions,
held-out MC oracle estimates, screenshots and GIFs, with zero behavioral training.
Original CORA benchmark confidence remains diagnostic; conditional certification
is exercised separately in the known-SCM benchmark. See the ledger for remaining
requirements and external comparator compatibility.

## Application execution handover

See [docs/runbook.md](docs/runbook.md) for preflight, execution, resume and
saved-data analysis. The application entry point defaults to preflight and
requires `--run` to execute experiments:

```bash
.venv/bin/python scripts/prepare_experiments.py
.venv/bin/python -m cora.benchmarks.release
```

No new application baseline results have been generated for this handover.
Twenty application–learner interfaces are checked with abstract JAX tracing;
this does not execute numerical optimizer updates or establish convergence.
Published-comparator source and adaptation status are recorded in
[docs/external_comparators.json](docs/external_comparators.json).
