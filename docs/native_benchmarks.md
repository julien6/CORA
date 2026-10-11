# Native JaxMARL experiment contract

Current frozen/adaptive configs use genuine JaxMARL transitions/rewards from
revision `976aeb152cb184a5095021968bba94da96eb6394`. No upstream code is edited.
Core synthetic problems remain for known-ground-truth causal validation. Custom
grid application prototypes are legacy fixtures, excluded from current runs.

| Environment | Controlled agents | Objective and intervention | Learner interface |
|---|---|---|---|
| World Comm | Leader + three pursuers | Mean pursuer return against fixed local-observation food-seeking foragers; reassignment of pursuit-sector duty hints | 20 discrete codes: native leader action; follower code modulo 5 |
| SMAX 3m | Three allied marines | Mean ally return against native heuristic enemies; reassignment of focus-target duty hints | Native discrete movement/attack actions; legal masks exposed to Q rollout manager |
| OvercookedV2 cramped_room_v2 | Two chefs | Native shared delivery reward; ingredient/cooking and plating/delivery duty hints | Native six discrete actions; observation grids flattened |
| JaxNav | Three differential-drive robots | Mean native navigation reward; reassignment of native navigation goals | Native 15-action discrete variant, normalized LiDAR observations, polygon collision model |

Duty hints are per-agent one-hot observations, not new privileged physical
state. Shared network inputs are padded to the same observation length. Critics
may consume concatenated local observations. No simulator-global state is added
to an actor. World Comm's native communication is retained, with homogeneous
codes allowing duplicate follower movement actions. The fixed opposing team is
part of the experimental context, not a learned competitive baseline.

Except for JaxNav goal delegation, duty assignment changes policy input rather
than restricting legal simulator actions. Policies can ignore these hints.
No task-fulfillment or non-degradation guarantee follows from their presence.
Training begins in the current organization; evaluating unseen assignments can
shift the continuation-policy distribution. Such shift must be studied, not
assumed harmless. Native randomness and partially observed interactions remain.

Every unordered controlled-agent pair defines an elementary responsibility swap;
compound transactions apply in ascending pair-index order and validate only the
final atomic assignment. Missions have exactly one assigned executor and all
executors are permitted to serve every current mission. JaxNav maps original
native goals through the new assignment at adoption/reset. Organization versions
persist through training episode resets. Evaluation uses native `step_env` without
autoreset, freezes after termination and zeros subsequent rewards.

The conservative dependency family contains all interactions among the configured
edits. This avoids silently asserting native causal locality, but offers no
structural parameter reduction: it is a correctness-first starting point for
future justified support graphs. Four pursuers give six edits/64 candidates;
SMAX and JaxNav have three edits/eight candidates; Overcooked has one edit/two
candidates and cannot test compound interaction advantages. Baseline budgets,
MC references and uncertainty are recorded; confidence remains diagnostic.

IPPO/MAPPO/IQL/VDN/QMIX reuse upstream trainer/network sources with environment
injection. Actor-count minibatching is environment-specific. SMAX native legality
masks reach the Q rollout manager and frozen Q actor; current PPO actors do not mask native actions,
so native handling of unavailable actions remains part of that adapted protocol.
Scalar simulator telemetry is broadcast per actor for upstream MPE-style loggers;
raw evaluation rewards and states remain unmodified. All algorithms share these
adapters within an environment. These are adapted
baseline executions, not faithful reproductions of every published protocol.

Native state trees are stored as lossless NPZ leaves, including nested chef and
SMAX opponent states. Rendering reconstructs those states with the same factory
and uses the upstream visualizers. SMAX unit snapshots suppress attack animation;
no intermediate physics ticks or opponent action sequences are invented. Rendering
and statistical analysis do not run policies, transitions, or training.

## Commands

```bash
# Safe preparation only.
.venv/bin/python -m cora.benchmarks.release
.venv/bin/python scripts/prepare_experiments.py
# Optional future operational smoke: random policies, no optimizer updates.
.venv/bin/python -m cora.benchmarks.release --config configs/native_smoke.yaml --run
# Full experiments: explicitly launch later.
.venv/bin/python -m cora.benchmarks.release --config configs/applications.yaml --run
.venv/bin/python -m cora.benchmarks.release --config configs/applications_adaptive.yaml --run
```

The no-training interface tests cover each native transition against its raw
counterpart, organizational adoption/goal semantics, state serialization and
rendering, plus twenty environment/learner combinations via `jax.eval_shape`.
They establish technical compatibility, not learned performance, convergence,
sample efficiency, causal-graph correctness or publication readiness.
