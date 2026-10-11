# CORA specification audit and engineering decisions

Authoritative source: `work_in_progress_document.pdf`, 11 October 2026.
This ledger distinguishes implemented mechanisms from unverified research claims.
Short local checks are authorized; expensive training and final evaluation are not run.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Eq. 1 atomic validity | organization.py, interventions.py; ordered swaps, post-validation | Small language implemented |
| Eq. 2 controlled context and continuation | rollout.py, simulation.py; fixed reset and policy metadata | Frozen protocol implemented |
| Eq. 3 exact Mobius / counterexamples | features.py and estimator tests | Implemented |
| T1 explicit temporal ancestry | temporal.py, benchmarks/temporal_trial.py | Implemented for known reservoir SCM; not learned graphs |
| T2 rank and target identifiability | estimator.py, benchmarks/methods.py | Implemented |
| Eq. 5 fixed design confidence | reliability.fixed_design_bounds | Implemented, caller establishes independence and noise bound |
| Eq. 6 adaptive confidence | reliability.adaptive_bounds, predictable-design coverage test | Implemented conditional API; `cora_conditional` integrates known bounds; original `cora` remains diagnostic |
| Eqs. 7–10 combined bounds | reliability.conservative_decision | Implemented conditional API, `cora_conditional` integrates all three bounds; arbitrary misspecification bounds unresolved |
| Eq. 11 joint-policy shift | reliability.policy_shift_bound | Implemented, joint TV must be supplied as a valid bound |
| Eq. 12 acquisition | benchmarks/methods.acquisition; direct inverse regression test | Exact finite-set implementation |
| Level 1 scale to 64 edits | benchmarks/scalable.py | Sampled candidate backend implemented; global optimization gap unknown |
| Level 2 shared resources and prerequisites | benchmarks/domains.py | Four small scripted application prototypes; realistic dynamics incomplete |
| Level 3 organization-conditioned MARL | learners.py, checkpoints, smoke tests | Five learners operational; new domains not integrated with MARL |
| Adaptive causal estimates | temporal.PolicyVersionedData, adaptation.py | Version-separated Algorithm 1 orchestration; injected trainers/trials, end-to-end adaptive MARL experiment remains |
| Published baseline identities | literature shortlist below | Candidates identified; faithful reproduction remains |
| Metrics and statistics | statistics.py, runner.py | Executed-nonnoop harm denominator corrected; broad adaptation/profile metrics remain |
| Figures/replays | media.py, scripts/validate_extended.py | Existing suite plus small domain artifacts |
| Final claims / hypothesis evidence | None | Requires larger independent experiments; not claimed |

## Appendix B decisions

1. Known-SCM validation uses two reservoirs, initial zero state, geometric decay,
   intervention-controlled inflow, discounted per-reservoir reward. Optional
   cross-inflow creates shared causal ancestry. Gaussian scalar observation noise
   has a known configured sub-Gaussian scale. No outcome is conditioned on a
   post-intervention covariate. The oracle is evaluator-only.
2. Transactions use ascending-index ordered swaps with atomic final validation.
3. Current elementary edits are responsibility swaps; more edit types require
   domain validators and tests. No claim of full deontic language implementation.
4. Public learner metadata excludes coefficient values and oracle returns.
5. Certification is conditional on exact features and known bounds. Diagnostic
   residual/leverage scales are never labeled calibrated. Misspecification needs
   externally justified structural bounds; guessed graphs cannot supply them.
6. Context and policy-version changes create separate datasets. No silent reuse.
7. Existing comparator families share query budgets; oracle is evaluation-only.
8. Decentralized actions and centralized atomic adoption are the current scope.
   Consensus is explicitly future work in Section 15, not a current requirement.

## External baseline literature audit

Candidate compositional work: Pruthi and Jensen, *Compositional Models for
Estimating Causal Effects*, arXiv:2406.17714v2 (2025).
https://arxiv.org/html/2406.17714v2
The paper models structured units with shared binary treatment and modular
component outcomes. CORA's compound edit vector is a different estimand. Its
hierarchical/additive models cannot be called faithful CORA baselines merely by
renaming the existing monomial ridge estimator. A component-level dataset and
explicit treatment mapping are necessary for reproduction.

Candidate sequential composition: *Structured Learning of Compositional
Sequential Interventions*, NeurIPS 2024.
https://proceedings.neurips.cc/paper_files/paper/2024/hash/d10c7e24c96db4b222688efd11b02940-Abstract-Conference.html
Requires multiple units and sequential interventions, unlike the single-context
binary edit-set runner. Review full text and released code before implementation.

Candidate organizational planner: *Multiagent Metareasoning through
Organizational Design*, AAAI 2014.
https://ojs.aaai.org/index.php/AAAI/article/view/8892
Explicit organizational design under a Dec-MDP with reasoning costs. Requires
an organizational decision-state model; the current singleton heuristic is not
this published algorithm.

These are audited candidates, not completed reproductions or confirmed nearest
prior work. Unavailable registry entries stay unavailable to avoid mislabeling.

## Local execution

`.venv/bin/python -m pytest -q tests`

`.venv/bin/python scripts/validate_extended.py`

The extended validation runs no behavioral training: 12-step scripted trajectories,
8 MC oracle replicas, 4/8 intervention budgets, one seed per domain. Its plots
are technical diagnostics, not aggregate performance evidence or preregistered
hypothesis tests. Algorithm mechanism reuse is not learned-weight transfer.

## Modular architecture adaptation

`compositional_modular` adapts the additive parallel architecture in Pruthi/Jensen
Appendix J to CCOR. Maximal public hyperedges define component inputs; small
independent MLP outputs are summed and trained end-to-end against observed scalar
returns. It uses uniform query selection and the same 40 optimizer steps per
fit as the global neural estimator. The source paper's shared binary treatment
and structured units are replaced by local edit subsets. This is a documented
architecture adaptation, not a faithful reproduction or the source paper's
identification guarantee. It is runnable, tested, and intentionally separate
from the unavailable `published_compositional` reproduction slot.

`cora_conditional` is an opt-in runner mode requiring explicit
`confidence_assumptions`, candidate-wise `structural_error_bounds` and
`policy_shift_error_bounds`. It logs assumptions and all three bound components.
The API does not establish hypotheses from data. The example noisy application
prototypes do not provide these justifications, so their run skips this mode.

## Current local evidence

`outputs/extended-smoke/` contains 48 final-budget application rows, four
known-SCM trials, eight 12-frame GIFs, eight replay screenshots, and a diagnostic
regret plot. No behavioral training was run by this command.
Application dependency families now derive from explicit conservative temporal
support graphs. Shared transitions give dense two-edit support; sparsity is not
assumed from symbolic separation. Known reservoir dynamics provide a controlled
case with sparse ancestry and a coupled counterexample.

`benchmarks/evaluation.py` defines sustained recovery, fixed-threshold performance
profiles, structure precision/recall and simultaneous coverage. These utilities
are tested; the current smoke report does not establish recovery or transfer
claims. Coverage tests are regression evidence under specified Gaussian noise,
not proofs that assumptions hold in an arbitrary application.

Remaining release blockers: faithful external comparator reproductions; direct
MARL adapters and adaptive-policy evaluation in the application prototypes;
expanded edit/deontic language; comprehensive metrics/figures and cross-domain
transfer experiments; research-scale statistical evidence. No all-requirements
completion claim is made.

## Execution handover update

`benchmarks/domain_env.py` now exposes functional JAX reset/step/local observations
and auto-reset with persistent organizational assignments for all four abstract
applications. `learners.prepare_training` constructs upstream trainers without
executing them. Abstract compilation tests cover all 20 application–learner
combinations; numeric training/evaluation remains unexecuted for this handover.

`benchmarks/release.py` wires frozen and explicit between-decision retraining
protocols, independent per-method adoption sequences, checkpoint loading,
intervention estimation, held-out MC references, raw traces, resume and analysis.
`domain_media.py` creates PNG/PDF charts, GIFs and matched replay comparisons
from saved data only. Configurations and exact future commands are documented
in `runbook.md`; preparation does not execute experiments.

`symbolic.py` adds typed immutable role/assignment/norm/relation transactions,
coverage and prerequisite checks, and stale-version rejection. These are a
generic host-side layer; application MARL candidates still use swaps. Normative
validation checks structural obligations and permissions, not realized temporal
mission fulfillment or unrestricted formal-contract compliance.

The external compositional source is now identified and pinned; metadata is in
`external_comparators.json`. Its original dependency/data contract differs from
CCOR. Faithful original experiment reproduction remains incomplete. The generic
MDP comparator is runnable and explicitly distinct from a published-method
reproduction. This ledger continues to distinguish runnable experimental wiring
from full draft compliance and publication evidence.

## Native benchmark migration (user-directed)

The user requested native JaxMARL environments for baselines/experiments instead
of the unclear custom application prototypes. Current sequential configs and
release execution now use World Comm, SMAX, OvercookedV2 and JaxNav. This changes
the empirical application plan, not the generic CORA decision rule. The earlier
warehouse/rescue/wildfire/hospital models and their historical results remain
legacy artifacts, not native equivalents and not current default experiments.
Exact adapter objectives, action mappings, assignment semantics, conservative
complete interaction support, and the two-agent kitchen's singleton limitation
are documented in `native_benchmarks.md`. No training/baseline experiments were
launched during migration; interface checks use native single transitions and
abstract trainer tracing. The original draft's compound-efficiency hypothesis
still needs controlled experiments with justified sparse support graphs.
