# CORA final-state demonstration

This folder is isolated from the research experiment outputs. All demo code,
configuration, raw results, visualizations, logs, and verification live here.
Existing environment implementations are imported read-only from the repository.

**No training occurs. These are not trained baseline results or predictions of
what IPPO, MAPPO, IQL, VDN, QMIX, or CORA will achieve.**

The requested training shortcut is implemented by supplying a fixed handwritten
controller in place of a learned continuation policy. All five learner names
refer to the same policy surrogate, so differences between actual
learning algorithms cannot be inferred. No neural checkpoint is fabricated.

## What runs

- Four exact synthetic scenarios: sparse, dense, incomplete graph, policy shift.
- A 64-edit sampled-candidate synthetic problem (pool-relative optimum only).
- Two known temporal SCMs: separated and causally coupled reservoirs.
- Native MPE Simple World Comm replaces Simple Spread in the current gallery.
- SMAX (3m), OvercookedV2 (cramped_room_v2), and JaxNav (BlankCross2) run
  genuine upstream dynamics, with versus reference controllers.
- JAX warehouse, rescue, wildfire, and hospital abstract environments, with
  mission-conditioned movement, heterogeneous service rates, stochastic dynamics,
  shared resources and agent failure. Their physical models are unchanged.

Three seeds are used for the original CORA demonstration. The additional native
JaxMARL visual demonstrations use a single predeclared seed (7) and separate
metrics; they are not integrated CORA baselines. World Comm is competitive, so
its pursuer and forager returns are reported separately. OvercookedV2 uses a
privileged full-state task planner; no decentralized learning claim is made.

For the original CORA demonstration, Every spatial reference uses four Monte Carlo replicas
per admissible intervention. The rendered seed is fixed beforehand to 7; it is
not chosen for favorable behavior. Held-out replays use stream 900000, separate
from reference-selection streams 100000–100003. This is a small demonstration,
not statistically adequate research evaluation.

Five organization profiles replace learning/estimation decisions:

| Profile | Choice | Privileged selection? |
|---|---|---|
| status_quo | No reorganization | No |
| fixed_swap | First admissible singleton edit | No |
| random_admissible | Seeded random admissible compound | No |
| singleton_reference | Best no-op/singleton by reference net value | Yes |
| compound_reference | Best admissible compound by reference net value | Yes |

Reference regret is computed against the same exact/MC candidate reference used
by the last two profiles. Their low reference regret is therefore by construction,
not an empirical CORA advantage. A selected compound can tie the status quo,
fail to help on a held-out rollout, or face unsolved structural limitations.
In particular, permutations cannot restore the capability of a failed agent.

The difference between idealized full and singleton reference selection can
illustrate interaction effects, but it does not establish sample efficiency,
uncertainty calibration, learned policy quality, adaptive training, causal graph
recovery, safety guarantees, or cross-domain learned-weight transfer.

## View and reproduce

Open `index.html`. It includes matched side-by-side environment animations,
individual replays, quantitative diagnostic plots, and links to raw files.
Figures and GIFs carry explicit no-training labels.

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/run.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/run.py --verify-only
```

`manifest.json` records controller identities, seeds, source hashes and zero
training/gradient updates. `verification.json` checks coverage, finite metrics,
admissible selections, and animated replay lengths. `results/results.csv` and
`results/results.json` contain measured outcomes. Native simulator
states, actions, rewards and assignments are saved in per-profile NPZ traces.

Time/RAM are still needed for simulation, compilation and rendering. Avoiding
training does not mean zero computation. The demo uses CPU and small horizons.
An accidental call to the imported training adapter raises an error.

`mock_catalog.json` maps all configured organizational baseline names to explicit
visualization fixtures. These names do not denote executed methods. No distinct
IPPO/MAPPO/IQL/VDN/QMIX performance or learned weights are fabricated.

## live organizational adoption

`adaptive/index.html` adds four matched online disruption/reconfiguration
comparisons. A predefined valid swap commits one step after the failure. The
same handwritten policy remains frozen; this replaces adaptive policy training
with a fixed fixture, rather than claiming to reproduce learned adaptation.
Organization versions and per-step assignments are recorded. These additional
runs have 24 rows, 24 traces and 12 replays, independently verified.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/adaptation.py
```

Rebuilding plots from existing static data requires no simulation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/run.py --plots-only
```

## Revised visual environments

The domain renderer shows recorded agent positions, responsibilities, failures,
queues, completed tasks and resources, with domain-specific illustrated scenery.
Scenery is decorative: these remain abstract grid simulators. Both static and
live-adoption GIFs were refreshed from existing NPZ traces without changing
trajectories or rewards. `visual_refresh_verification.json` records that check.

Open `library_environments/index.html` for World Comm, SMAX, OvercookedV2 and
JaxNav side-by-side GIFs and initial/middle/final screenshots. The kitchen uses a contextual chef/station renderer with additional original
native-renderer PNG screenshots (`native_frame_*.png`). Native state leaves,
actions (zero-padded, with original sizes documented), transition keys and actual
per-agent rewards are saved in each `trace.npz` with a schema in `metadata.json`.
SMAX images show decision-step states rather than interpolated physics ticks.
All episodes stop collecting rewards at termination and freeze the final state;
no automatic reset is inserted. A single episode cannot establish performance.

```bash
# Render saved application traces; no simulator runs or training.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/refresh_visuals.py
# Run short native episodes, without training.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/library_demo.py
# Rebuild the main gallery after adding native demonstrations.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/run.py --plots-only
# Check saved native traces and GIFs.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python demonstrations/scripted_final_state/library_demo.py --verify-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider demonstrations/scripted_final_state/test_demo.py
```

Simple Spread artifacts are retained only in `legacy_simple_spread/`, with their
old rows explicitly archived. Current core results contain 165 rows across 11
conditions; the eight new native rollouts have a separate results file and
provenance. Nothing is written to the research baseline outputs or paper.

The native OvercookedV2 renderer needs `imageio`; it is pinned in
`render-requirements.txt` and installed locally in this folder's `.deps/`:

```bash
.venv/bin/python -m pip install --no-deps --target demonstrations/scripted_final_state/.deps -r demonstrations/scripted_final_state/render-requirements.txt
```

JaxNav uses the upstream polygon-agent singleton map and normalized observations.
The pinned JaxMARL revision has unsupported collision handling for the singleton's
circle-map default and an uninitialized observation field in the unnormalized
branch; selecting supported native options avoids editing third-party code.
`rerender_native.py` refreshes World Comm/SMAX from saved native states with no
new episode. Main and native verification files are intentionally separate.
