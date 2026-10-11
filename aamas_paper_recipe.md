# Recipe: from ideation to submission of an AAMAS paper

> **How to use.** Duplicate this file for each paper (e.g. `recipe_<ACRONYM>.md`).
> Replace the `{{…}}` fields, tick the boxes as you go, and keep the file
> in the project repository: it becomes the paper's logbook.
> **Gates** are checkpoints: move on to the next phase only if the criterion is met;
> otherwise pivot or reduce the scope.
>
> 📌 **Paper appendix rule.** The paper may come with as many appendices as needed,
> but **only the main body of the paper (references included, appendices excluded) is reviewed**. Appendices provide
> additional details; they must contain **no information that is necessary** for the introduction,
> for understanding the contribution, or for assessing the claims. Standing test: *can a reviewer who
> never opens the appendices understand, judge and be convinced?* See the section
> [Paper appendix policy](#paper-appendix-policy) in Phase 10.
>
> ⚠️ Do not confuse the **paper's appendices** (submitted with the paper) with
> **Annexes A–L of this document**, which are internal working templates.

---

## Project identity card

| Field | Value |
|---|---|
| Working title | {{title}} |
| Contribution acronym | {{ACRONYM}} |
| Target venue / track | {{AAMAS 20XX – main track / Blue Sky / workshop}} |
| Area / sub-community | {{e.g. Learning and Adaptation}} |
| Abstract deadline | {{date}} |
| Paper deadline | {{date}} |
| Author response phase | {{dates}} |
| Authors and roles | {{name – role}} |
| Repository (private) | {{link}} |
| Anonymized repository | {{link}} |
| Initial informal document | {{link}} |

---

## Contents

0. [Scoping and backward planning](#phase-0--scoping-and-backward-planning)
1. [Mapping the community](#phase-1--mapping-the-community)
2. [Encyclopedic review of the field](#phase-2--encyclopedic-review-of-the-field)
3. [Confronting the topic with the literature](#phase-3--confronting-the-topic-with-the-literature)
4. [Hypotheses, contributions and name](#phase-4--hypotheses-contributions-and-name)
5. [Designing the contribution](#phase-5--designing-the-contribution)
6. [Claims](#phase-6--claims)
7. [Deductive validation](#phase-7--deductive-validation)
8. [Experimental protocol](#phase-8--experimental-protocol)
9. [Execution, results and discussion](#phase-9--execution-results-and-discussion)
10. [Writing](#phase-10--writing)
11. [Adversarial review](#phase-11--adversarial-review)
12. [Submission](#phase-12--submission)
13. [After submission](#phase-13--after-submission)
- [Annexes: templates to fill in](#annexes-templates-to-fill-in)

---

## Phase 0 — Scoping and backward planning

**Goal**: set the course before investing time and compute.

### Actions
- [ ] Read the CfP in full and note:
  - [ ] format (ACM sigconf), page limit, whether references count toward it
  - [ ] double-blind review and anonymization rules
  - [ ] abstract deadline distinct from the paper deadline
  - [ ] policy on supplementary material, appendices and code (where to put them, in the PDF or separately; reminder: only the main body + references is reviewed)
  - [ ] policy on LLM use
  - [ ] existence and dates of the author response phase
  - [ ] available tracks (main, Blue Sky, JAAMAS, demo, workshops)
- [ ] Choose the target: main track, or a workshop if the work is preliminary (ALA, EMAS, COINE, MABS…)
- [ ] Build the backward plan (see table below)
- [ ] Estimate the compute budget: `seeds × environments × (method + baselines + ablations) × run duration`

### Typical backward plan

| Milestone | Target date |
|---|---|
| Mapping and review completed | {{D-…}} |
| Gaps, hypotheses and claims frozen | {{D-…}} |
| Feasibility pilot conclusive | {{D-…}} |
| Implementation complete and tested | {{D-…}} |
| Experimental protocol frozen | {{D-…}} |
| Expected-results document frozen (before any baseline run) | {{D-…}} |
| Final experiments completed | ~D-21 |
| Full draft written | ~D-14 |
| Mock reviews returned | ~D-10 |
| Abstract and keywords submitted | {{abstract date}} |
| Submission | D-1 at the latest |

### Deliverable
- [ ] Identity card above filled in

### ✅ Gate 0
- [ ] The compute budget covers the minimal credible protocol (otherwise: reduce the scope **now**).

---

## Phase 1 — Mapping the community

**Goal**: know who you are talking to, in which language, and against whom you are positioning yourself.

### Actions — building the graph
- [ ] Select 15 to 30 "seed" papers very close to the topic
- [ ] Extract authors, co-authors and references (DBLP, Semantic Scholar API, OpenAlex)
- [ ] Build the co-authorship / co-citation graph (networkx, VOSviewer, Connected Papers, Litmaps)
- [ ] Identify clusters, central nodes and "bridges" to other communities
- [ ] Cross-reference with recent AAMAS PC / SPC / area chairs, organizers of related workshops, JAAMAS / JAIR / AIJ editors
- [ ] Extend to relevant sister venues (NeurIPS, ICML, ICLR, TMLR, IJCAI, AAAI, KR, CoRL, JASSS…)

### Actions — what to extract from it
- [ ] Canonical vocabulary (→ glossary, Annex B)
- [ ] Expected formalisms (Dec-POMDP, POSG, Markov games, BDI, Moise/AGR, Petri nets, epistemic logics…)
- [ ] Unavoidable foundational papers ("must-cite")
- [ ] Recent reference papers (last 2–3 years)
- [ ] Open debates and groups' positions
- [ ] Standard benchmarks and protocols of the sub-community

> ⚠️ **Ethics and effectiveness.** The graph is used to speak the community's language and to choose
> the area and keywords well (the legitimate lever for reviewer assignment), **not** to target reviewers.
> Gratuitous citations of PC members are noticeable and hurt the paper.

### Deliverables
- [ ] Annotated graph ({{file link}})
- [ ] Must-cite list (Annex B)
- [ ] Terminology glossary (Annex B)

---

## Phase 2 — Encyclopedic review of the field

**Goal**: understand current challenges and perspectives, not just catalogue works.

### Actions
- [ ] Define reproducible queries and inclusion / exclusion criteria
- [ ] Forward / backward snowballing from the seed papers
- [ ] Skim 80 to 150 papers, read 30 to 50 in depth
- [ ] Fill in a reading sheet for each paper read in depth (Annex C)
- [ ] Read recent surveys and position papers
- [ ] Extract the "limitations / future work" sections of the most cited papers (gaps self-declared by the community)
- [ ] Go through recent tutorials, keynotes and Blue Sky papers (medium / long-term perspectives)
- [ ] Build a taxonomy along 3 to 5 relevant axes

### Deliverables
- [ ] Taxonomy of approaches
- [ ] Comparative table of approaches
- [ ] "State of the art + open questions" synthesis (3–5 pages)

---

## Phase 3 — Confronting the topic with the literature

### 3.1 Reformulating the problem
- [ ] Confront the informal document with the taxonomy: already solved? partially? ill-posed? phrased differently elsewhere?
- [ ] Assess:
  - **Alignment**: {{is the community interested in it now? evidence}}
  - **Originality**: {{difference from the closest work}}
  - **Impact**: {{who benefits, what does it unlock?}}
- [ ] Reformulate the problem in the community's vocabulary

> **Problem in one sentence**: {{…}}
>
> **Problem in one paragraph**: {{…}}

- [ ] Novelty test: actively search for the paper that would "kill" the idea (recent arXiv, workshops, theses)
  - Result: {{none / paper X → how we differ}}

### 3.2 Decomposition into gaps
- [ ] Define G1…Gn (typically 2 to 4), each:
  - verifiable in the literature,
  - independent of the others,
  - together covering the overall problem.

| Gap | Statement | Evidence of the gap (references) |
|---|---|---|
| G1 | {{…}} | {{…}} |
| G2 | {{…}} | {{…}} |
| G3 | {{…}} | {{…}} |

- [ ] Fill in the related work × gaps matrix (Annex D)

### ✅ Gate 3
- [ ] Novelty confirmed (no existing work covers all the gaps)
- [ ] At least two credible and reproducible external baselines identified
- Otherwise: reformulate or pivot.

---

## Phase 4 — Hypotheses, contributions and name

### Actions
- [ ] For each Gi, formulate one or more falsifiable hypotheses Hi ("if X, then Y, under conditions Z")
- [ ] Associate a sub-contribution Ci with each Hi (component, formalism, theorem, metric, environment…)
- [ ] Define the overall contribution (algorithm / method / framework) integrating the Ci
- [ ] Choose the name and acronym:
  - [ ] pronounceable
  - [ ] not already taken (check Google Scholar, GitHub, arXiv)
  - [ ] evokes the core idea
- [ ] Formulate the overall hypothesis H

> **Overall hypothesis H**: {{the combination of the Ci solves P better than existing work, because…}}

| Gap | Hypothesis | Sub-contribution |
|---|---|---|
| G1 | H1: {{…}} | C1: {{…}} |
| G2 | H2: {{…}} | C2: {{…}} |
| G3 | H3: {{…}} | C3: {{…}} |

### Deliverable
- [ ] First version of the traceability matrix (Annex E)

---

## Phase 5 — Designing the contribution

### 5.1 Informal ideation
- [ ] One-page description: intuition, core idea, toy example

### 5.2 Component description
- [ ] For each Ci: role, inputs / outputs, interactions with the other components
  (without yet addressing auxiliary metrics / properties)

### 5.3 Feasibility pilot ("kill test")
- [ ] Rough version implemented on a toy environment
- [ ] Expected signal observed? {{yes / no – observations}}
- If not: go back to Phase 3 or 4.

### 5.4 Mathematical formalization
- [ ] Preliminaries: notation and base model taken from the literature
- [ ] Formal definition of the new objects
- [ ] Propositions / theorems with proofs: full statement, assumptions and intuition (or proof sketch) in the main body; full proof in the appendix
- [ ] Pseudo-code consistent with the notation, **in the main body** if the algorithm is the contribution
- [ ] All notation and definitions used in the main body are defined **in the main body** (never only in the appendix)

### 5.5 Schematization
- [ ] Overview figure (Figure 1, understandable without the text, self-contained caption)
- [ ] Per-component diagrams if needed

### 5.6 Implementation plan
- [ ] Requirements specification (lightweight SRS: 1 page)
- [ ] Technical survey of recent, well-established building blocks ({{e.g. JaxMARL, BenchMARL, EPyMARL, PettingZoo, TorchRL…}})
- [ ] Design: class diagram + sequence diagram (lightweight UML)
- [ ] Implementation of the contribution
- [ ] Implementation of the auxiliary metrics / properties needed for the claims
- [ ] Unit tests on the mathematical invariants (the code does implement the formalism)

> 💡 For research code, favour lightweight design and invariant tests
> over exhaustive UML documentation.

### ✅ Gate 5
- [ ] The pilot shows a signal
- [ ] The implementation passes the invariant tests

---

## Phase 6 — Claims

### Actions
- [ ] Formulate numbered claims, each linked to at least one gap and measurable (Annex F)
- [ ] **Freeze claims and protocol before the final experiments** (dated internal pre-registration in the repository)
  - Freeze date: {{date}} – commit: {{hash}}
- [ ] For each claim, plan what will be written if it is not supported
- [ ] Translate each claim into quantitative expectations in the expected-results document (Phase 8, Annex L)

> Example of a well-formed claim: *"Cl2: {{ACRONYM}} achieves a higher return than the best baseline
> (IQM, 95% CI) on at least 4 of the 6 {{benchmark}} scenarios, at equal sample budget."*

---

## Phase 7 — Deductive validation

### Actions
- [ ] Establish the relevant properties:
  - [ ] time complexity
  - [ ] memory complexity
  - [ ] convergence
  - [ ] bounds (regret, approximation error, suboptimality…)
  - [ ] safety / consistency guarantees
  - [ ] game-theoretic properties (equilibria, incentives)
- [ ] Compare with the baselines' properties ("theoretical guarantees" table)
- [ ] Derive testable predictions (trends, orders of magnitude, expected intervals)
- [ ] Make the theorems' assumptions explicit, as well as the gap with the experimental setting
- [ ] Place each result right after the definition of the object concerned; full proofs in the appendix
- [ ] The main body alone must be enough to believe the result: precise statement, explicit assumptions, key idea of the proof and its consequence for the claims. The appendix is only for detailed verification.

### Guarantees table

| Property | {{ACRONYM}} | Baseline A | Baseline B | Baseline C |
|---|---|---|---|---|
| Time complexity | {{…}} | {{…}} | {{…}} | {{…}} |
| Memory complexity | {{…}} | {{…}} | {{…}} | {{…}} |
| Convergence | {{…}} | {{…}} | {{…}} | {{…}} |
| Bound | {{…}} | {{…}} | {{…}} | {{…}} |

### Predictions to verify empirically

| Prediction | Source (Prop./Thm.) | Verified by (Exp.) | Result |
|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} |

> These predictions are carried over into the expected-results document (Phase 8, Annex L),
> where they are turned into numerical values and intervals per baseline and per scenario.

---

## Phase 8 — Experimental protocol

Dedicated section of the paper. Fill in the template in Annex G.

### Checklist
- [ ] Research questions (RQs) explicitly linked to the claims
- [ ] Recognized environments / benchmarks, with justification
- [ ] Specific environment (if needed) described and released
- [ ] External baselines taken from the Phase 3 matrix, official implementations where possible
- [ ] **Same hyperparameter tuning budget for all methods**
- [ ] Internal baselines and ablations (each ablation answers a precise question)
- [ ] Standard metrics + specific metrics formally defined
- [ ] Statistical protocol:
  - [ ] sufficient number of seeds (≥ 10 if possible, rarely < 5)
  - [ ] robust aggregation (IQM, bootstrap CIs, performance profiles – e.g. `rliable`, Agarwal et al., 2021)
  - [ ] appropriate tests (Welch, Mann-Whitney) + multiple-comparison correction (Holm)
  - [ ] in cooperative MARL: follow the protocol of Gorsane et al. (2022)
- [ ] Reproducibility:
  - [ ] hardware (CPU, GPU, memory)
  - [ ] software (versions, seeds)
  - [ ] total compute time
  - [ ] full hyperparameter table (appendix)
  - [ ] **anonymized** repository (Anonymous GitHub or archive in the supplementary material)
- [ ] Threats to validity: internal, external, construct
- [ ] Main body / appendix split checked:
  - **Main body**: RQs, environments, baselines, ablations, metrics, statistical protocol, tuning budget, number of seeds, summary of hardware and compute time, link to the anonymized repository
  - **Appendices**: full hyperparameter tables, baseline implementation details, exhaustive scenario descriptions, scripts and configurations

### Expected-results document (pre-registered predictions)

Once the contribution is implemented and the protocol is fixed, and **before running any baseline**,
write an internal document (template in Annex L) that describes every baseline and predicts its numerical results
as realistically as possible. Reasoning from the hypotheses (assumed true), the theoretical results, and the
experimental biases you can foresee, you can usually anticipate trends and sometimes fairly precise values.
This document is the reference against which the real results will be compared in Phase 9.

**Contents**
- [ ] **Baseline descriptions**, for every condition that will be run: our contribution, each external baseline, each internal baseline / ablation. For each one: source and implementation (official or reimplemented), configuration, tuning budget, gaps covered, known strengths and weaknesses.
- [ ] **Expected numerical results** for each baseline × scenario × metric:
  - point estimate (expected mean or IQM),
  - prediction interval (e.g. 80%) and expected spread across seeds,
  - expected rank among the conditions,
  - rationale and source of the prediction: hypothesis Hi, theorem / proposition, values reported in the original paper, pilot results, extrapolation,
  - confidence level (low / medium / high).
- [ ] **Expected trends and orderings**: ranking of methods, scaling behaviour (e.g. with the number of agents), learning-curve shapes, ordering of ablations by expected performance drop, expected trade-offs (performance vs. compute, etc.).
- [ ] **Anticipated experimental biases**, each with its expected direction and rough magnitude, for example:
  - reimplemented baselines underperforming the values reported in their original papers,
  - unequal tuning effort despite equal budgets (familiarity with one's own method),
  - differences in environment versions, evaluation protocol (number of evaluation episodes, deterministic vs. stochastic policies) or aggregation,
  - compute or time limits truncating training,
  - seed variance and sensitivity to initialization,
  - hardware / numerical non-determinism.
- [ ] **Bias-adjusted expectations**: for each value, give both the "ideal" prediction (hypotheses true, no bias) and the "realistic" prediction (adjusted for the anticipated biases).
- [ ] **Link to the claims**: for each claim, the expected values that would support it and the values that would refute it.

**Rules**
- [ ] Freeze the document (dated commit) **before** running the baselines; any later change goes into a dated addendum with its justification, never into an edit of the original values.
- [ ] The predictions must not influence how the experiments are run: no stopping the tuning of a baseline once it reaches its predicted value, no extra seeds only where results disappoint, no dropping of scenarios after the fact.
- [ ] Predictions are honest estimates, not targets: a well-calibrated document includes cases where our contribution is expected to be weaker.

### ✅ Gate 8
- [ ] A colleague outside the project can reproduce the experiment from the written protocol alone.
- [ ] The expected-results document is complete and frozen before any baseline run.

---

## Phase 9 — Execution, results and discussion

### Actions
- [ ] Rigorous run tracking (W&B / MLflow / Hydra), raw results saved
- [ ] Organize the Results section **by RQ**, not by environment
- [ ] For each RQ: figure/table → factual observation → explicit verdict on the claim
- [ ] Confront empirical results with theoretical predictions (Phase 7)
- [ ] Analyze the ablations: necessary / marginal / redundant components
- [ ] At least one qualitative analysis (learned behaviours, visualizations, failure cases)
- [ ] Discuss biases and scope of the results
- [ ] Wrap-up: claims supported → gaps bridged → H supported *under the tested conditions*
- [ ] Each claim is decided by a figure or table **in the main body**; the appendices only contain complementary results (additional scenarios, per-seed curves, sensitivity analyses) that confirm without being indispensable

### Expected vs. observed comparison

Once all baselines have actually been run and all numerical results are available, compare them
systematically with the frozen expected-results document (Annex L, "Observed" columns and comparison section).

- [ ] Fill in the observed value and interval next to each expected value (baseline × scenario × metric)
- [ ] Classify each result:
  - **Confirmed**: inside the realistic prediction interval
  - **Deviation explained by an anticipated bias**: outside the interval, in the direction and of the magnitude foreseen for a listed bias
  - **Unexpected deviation**: neither of the above
- [ ] For every unexpected deviation, **investigate before interpreting**: implementation bug, configuration error, protocol mismatch, environment version, evaluation code. Only once these are ruled out is the deviation treated as a genuine finding.
- [ ] Check the expected trends and orderings (ranking, scaling, ablation ordering) against the observed ones
- [ ] Compute simple calibration indicators: share of observed values inside the predicted intervals, rank correlation between expected and observed orderings (e.g. Kendall's τ), mean absolute deviation
- [ ] Never revise the original predictions after the fact; disagreements are documented, not erased
- [ ] Feed the analysis into the paper: the "theory vs. empirics" discussion, the threats to validity, and any genuine surprises (which are often the most interesting findings to report)
- [ ] Optional: mention in the paper (briefly in the main body, details in an appendix) that predictions were pre-registered internally before the baselines were run; this strengthens credibility but is not required to understand the contribution

### Verdict tracking

| Claim | RQ | Experiment | Expected (Annex L) | Observed | Verdict | Comment |
|---|---|---|---|---|---|---|
| Cl1 | RQ1 | {{…}} | {{…}} | {{…}} | {{supported / partial / not supported}} | {{…}} |
| Cl2 | RQ1 | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} |
| Cl3 | RQ2 | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} |

---

## Phase 10 — Writing

### Indicative page budget (8 pages)

| Section | Pages | Content | Status |
|---|---|---|---|
| Introduction | ~1.25 | context, problem, difficulty, gaps, idea, numbered contributions, key results | [ ] |
| Related Work | ~0.75 | organized by gap; ends with the positioning | [ ] |
| Preliminaries | ~0.5 | standard formalism, notation | [ ] |
| {{ACRONYM}} | ~2–2.5 | components, formalization, theorems, Figure 1, algorithm | [ ] |
| Experimental protocol | ~1 | structure from Phase 8 | [ ] |
| Results and discussion | ~1.5 | by RQ, ablations, theory vs. empirics | [ ] |
| Conclusion | ~0.3 | two paragraphs | [ ] |
| References | outside limit* | | [ ] |
| Appendices | unlimited* | additional details only — **not reviewed** | [ ] |

\* Check the exact rules of the edition in the CfP.

### Paper appendix policy

**Principle**: only the main body of the paper (references included, appendices excluded) is reviewed.
The main body must therefore be **self-contained**; appendices are a complement for readers who want to verify or reproduce.

| Must be in the main body | Can go in an appendix |
|---|---|
| Problem, motivation, gaps, contributions | Extended literature review |
| Formalism, notation and definitions used in the main body | Auxiliary notation used only in the appendix |
| Full theorem statements, assumptions, intuition / proof sketch | Full proofs, technical lemmas |
| Main algorithm (pseudo-code) | Variants, implementation details |
| Overview figure of the contribution | Additional detailed diagrams |
| Experimental protocol (RQs, benchmarks, baselines, metrics, statistics) | Full hyperparameters, configurations, baseline details |
| Results deciding each claim, main ablations | Additional results, sensitivity analyses, per-seed curves |
| Limitations | Secondary discussions, additional failure cases |

**Rules**
- [ ] No information necessary to understand the contribution or assess a claim is only in an appendix
- [ ] No "see appendix" pointer where the reader **needs** it to follow the reasoning; pointers are phrased as complements ("the full proof is given in Appendix B; the key idea is…")
- [ ] Each appendix is referenced at least once from the main body
- [ ] Appendices have their own explicit titles and can be read independently
- [ ] No circumventing the page limit: do not move essential content to the appendix to save space (reviewers notice and penalize it)
- [ ] Appendices respect anonymity just like the main body

**Read-without-appendix test** (to do in Phase 11): a reviewer reads only the main body and references,
then answers: *did I understand the contribution? can I judge each claim? am I missing a definition,
an assumption or a result?* Any negative answer requires moving the relevant content back into the main body.

### Structure of the introduction
1. Context and stakes
2. Precise problem
3. Why it is hard
4. Why existing work falls short (gaps G1…Gn)
5. Our idea in one or two sentences
6. Numbered contributions, with pointers to sections
7. Key quantitative results

### Writing rules
- [ ] Title and abstract contain the exact keywords of the sub-community (bidding)
- [ ] Notation consistent with the reference papers
- [ ] Balanced references: foundational + recent AAMAS/JAAMAS + sister venues
- [ ] Each claim in the introduction points to evidence (check with Annex E)
- [ ] Self-citations in the third person, anonymity preserved
- [ ] Figure 1 visible early (page 1 or top of page 2)

### Conclusion (two paragraphs)
- [ ] ¶1: summary of the problem, contribution and results
- [ ] ¶2: honest limitations; future work (with any preliminary results) tied to the community's current directions; long-term perspectives

---

## Phase 11 — Adversarial review

### Actions
- [ ] 2 to 3 mock reviews (at least one reviewer outside the project) using the form in Annex H, **providing only the main body and references**, as in a real review
- [ ] Read-without-appendix test (see Phase 10) passed: no indispensable information is only in an appendix
- [ ] "Reviewer 2" pass: list of likely objections defused (Annex I)
- [ ] Clarity pass: are the abstract + Figure 1 + end of the introduction enough to understand the contribution?
- [ ] Consistency pass: notation, G/H/C/Cl numbering, cross-references, captions
- [ ] Language proofreading

---

## Phase 12 — Submission

See the detailed checklist in Annex J.

- [ ] Abstract, title, area and keywords submitted by the intermediate deadline
- [ ] Conflicts of interest declared
- [ ] Final PDF compliant and anonymized
- [ ] Supplementary material uploaded
- [ ] Submission made at least 24 hours before the deadline

---

## Phase 13 — After submission

- [ ] Draft answers to anticipated objections (Annex I)
- [ ] Complementary experiments ready to launch
- [ ] Rebuttal written following the template in Annex K
- [ ] If rejected: incorporate the reviews and choose the next venue ({{IJCAI / NeurIPS / JAAMAS…}})

---

# Annexes: templates to fill in

## Annex A — Decision log

| Date | Decision | Rationale | Alternatives discarded |
|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} |

## Annex B — Mapping and glossary

### Must-cite papers

| Reference | Type (foundational / recent / survey) | Venue | Why unavoidable |
|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} |

### Key groups and researchers

| Group / researcher | Institution | Topic | Key papers | Link to our topic |
|---|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} | {{…}} |

### Glossary

| Community term | Definition | Avoid (non-standard term) |
|---|---|---|
| {{…}} | {{…}} | {{…}} |

## Annex C — Reading sheet (one per paper)

```markdown
### {{Authors, Year, Title}} — {{Venue}}
- **Problem**:
- **Formalism**:
- **Assumptions**:
- **Method**:
- **Benchmarks**:
- **Metrics**:
- **Key results**:
- **Stated limitations**:
- **Unstated limitations (critical reading)**:
- **Code available**: yes / no – link
- **Gaps covered**: G1 ● / G2 ◐ / G3 ○
- **Potential baseline**: yes / no – why
```

## Annex D — Related work × gaps matrix

Legend: ● fully covers, ◐ partially, ○ not at all.

| Work | G1 | G2 | G3 | Code available | Role (external baseline / discussion) |
|---|---|---|---|---|---|
| {{Method A (year)}} | | | | | |
| {{Method B (year)}} | | | | | |
| {{Method C (year)}} | | | | | |
| **{{ACRONYM}} (ours)** | ● | ● | ● | yes | — |

## Annex E — Traceability matrix

| Gap | Hypothesis | Contribution | Claim(s) | Deductive evidence | Inductive evidence | Main-body section | Appendix complements |
|---|---|---|---|---|---|---|---|
| G1 | H1 | C1 | Cl1, Cl2 | {{Prop. 1}} | {{Exp. 1, Fig. 3}} | {{§4.1, §6.1}} | {{App. B (proof), App. D}} |
| G2 | H2 | C2 | Cl3 | {{Thm. 1}} | {{Exp. 2, Tab. 2}} | {{…}} | {{…}} |
| G3 | H3 | C3 | Cl4 | {{—}} | {{Exp. 3, ablation}} | {{…}} | {{…}} |

> The "Main-body section" column must never be empty: each claim must be established in the main body.

## Annex F — Claim specification

| Claim | Statement | Gap(s) | Metric | Compared to | Condition | Success threshold | If not supported |
|---|---|---|---|---|---|---|---|
| Cl1 | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} |

## Annex G — Experimental protocol

```markdown
### Research questions
- RQ1 (Cl…):
- RQ2 (Cl…):

### Environments / benchmarks
| Environment | Scenarios | Justification | Reference |

### External baselines
| Method | Implementation (official?) | Gaps covered | HP tuning |

### Internal baselines / ablations
| Variant | Component removed/replaced | Question it answers |

### Metrics
| Metric | Definition | Standard / specific | Claim(s) |

### Statistical protocol
- Number of seeds:
- Aggregation:
- Confidence intervals:
- Statistical tests:
- Multiple-comparison correction:

### Reproducibility
- Hardware:
- Software and versions:
- Total compute time:
- Hyperparameters: (table in the paper's appendix)
- Anonymized repository:

### Threats to validity
- Internal:
- External:
- Construct:
```

## Annex H — Mock review form

```markdown
**Reviewer**:            **Date**:
**Instruction**: assess only the main body of the paper and the references, without opening the appendices.
**Summary of the contribution in 3 sentences**:

| Criterion | Score (1–5) | Comment |
|---|---|---|
| Relevance to AAMAS / the sub-community | | |
| Originality | | |
| Technical quality | | |
| Significance | | |
| Clarity | | |
| Reproducibility | | |

**Strengths**:
**Weaknesses**:
**Information missing from the main body (that I had to look for in the appendix)**:
**Questions to the authors**:
**Recommendation**: strong accept / accept / weak accept / borderline / weak reject / reject
**What would change my score**:
```

## Annex I — Likely objections ("Reviewer 2")

| Objection | Likelihood | Where defused in the paper? | Answer prepared for the rebuttal |
|---|---|---|---|
| Important baseline missing | | | |
| Unequal hyperparameter tuning | | | |
| Too few seeds / no statistical tests | | | |
| Environments too simple (toy) | | | |
| Unrealistic theorem assumptions | | | |
| Incremental novelty | | | |
| Compute cost not discussed | | | |
| Scalability (number of agents) | | | |
| Insufficient ablations | | | |
| Specific metric not justified | | | |
| Essential content relegated to the appendix (paper not self-contained) | | | |
| {{topic-specific objection}} | | | |

## Annex J — Submission checklist

### Format
- [ ] Official ACM template of the edition, margins / fonts unmodified
- [ ] Page limit respected
- [ ] Figures readable in black and white and at actual size
- [ ] All references complete and consistent

### Anonymization
- [ ] Names and affiliations removed
- [ ] Acknowledgements and funding removed
- [ ] PDF metadata cleaned
- [ ] Self-citations in the third person
- [ ] No link to a named repository or page
- [ ] Code and supplementary material anonymized

### Content
- [ ] Every announced contribution is delivered
- [ ] Every claim has evidence (Annex E up to date)
- [ ] All notation defined before use
- [ ] Limitations explicitly discussed

### Paper appendices
- [ ] The main body is self-contained: read-without-appendix test passed
- [ ] No indispensable definition, assumption or result only in an appendix
- [ ] Each appendix is referenced from the main body, with a pointer phrased as a complement
- [ ] Appendices placed where the CfP requires (in the PDF after the references, or as supplementary material)
- [ ] Appendices anonymized

### Platform
- [ ] Area and keywords chosen carefully
- [ ] Conflicts of interest declared
- [ ] Required declarations (LLM use, ethics, reproducibility)
- [ ] Supplementary material uploaded (proofs, hyperparameters, additional results, code)
- [ ] PDF downloaded back from the platform and checked after upload

## Annex K — Author response (rebuttal) template

```markdown
We thank the reviewers for their constructive feedback. We address the main concerns below.

**[R1, R3] {{Shared concern}}**
{{Factual, quantitative answer, with a pointer to the section / supplementary material.}}

**[R2] {{Concern}}**
{{…}}

**Planned changes for the camera-ready version**
- {{…}}
```

Principles: answer point by point, factually and courteously; prioritize the concerns
most likely to change a score; provide numbers rather than promises.

## Annex L — Expected-results document (pre-registered predictions)

> Freeze before running any baseline. Freeze date: {{date}} – commit: {{hash}}.
> Later changes go into the addenda section only.

### L.1 Assumptions behind the predictions
- Hypotheses assumed true: {{H1, H2, …}}
- Theoretical results used: {{Prop. 1, Thm. 1, …}}
- Other sources: {{values reported in original papers, pilot results, …}}

### L.2 Baseline descriptions

| Condition | Type (ours / external / internal-ablation) | Source & implementation | Configuration | Tuning budget | Gaps covered | Known strengths / weaknesses |
|---|---|---|---|---|---|---|
| {{ACRONYM}} | ours | {{…}} | {{…}} | {{…}} | G1, G2, G3 | {{…}} |
| {{Baseline A}} | external | {{official / reimplemented}} | {{…}} | {{…}} | {{…}} | {{…}} |
| {{ACRONYM}} w/o C1 | internal-ablation | {{…}} | {{…}} | {{…}} | {{…}} | {{…}} |

### L.3 Anticipated experimental biases

| Bias | Affected condition(s) | Expected direction | Expected magnitude | Mitigation in the protocol |
|---|---|---|---|---|
| Reimplementation underperforms reported values | {{…}} | ↓ | {{e.g. −5 to −15%}} | {{…}} |
| Unequal tuning effort | {{…}} | {{…}} | {{…}} | {{…}} |
| Evaluation protocol / environment version differences | {{…}} | {{…}} | {{…}} | {{…}} |
| Training truncated by compute limits | {{…}} | {{…}} | {{…}} | {{…}} |
| Seed variance | all | ± | {{…}} | {{…}} |
| {{other}} | {{…}} | {{…}} | {{…}} | {{…}} |

### L.4 Expected numerical results

One table per scenario (or per metric). "Ideal" = hypotheses true, no bias; "Realistic" = adjusted for the biases in L.3.

**Scenario: {{name}} — Metric: {{name}}**

| Condition | Ideal (point) | Realistic (point) | Realistic interval (80%) | Expected spread across seeds | Expected rank | Rationale / source | Confidence | Observed (filled after runs) | Status |
|---|---|---|---|---|---|---|---|---|---|
| {{ACRONYM}} | {{…}} | {{…}} | {{[…, …]}} | {{…}} | {{…}} | {{H1, Thm. 1}} | {{low / med / high}} | | {{confirmed / explained / unexpected}} |
| {{Baseline A}} | {{…}} | {{…}} | {{[…, …]}} | {{…}} | {{…}} | {{reported value − bias}} | {{…}} | | |
| {{ACRONYM}} w/o C1 | {{…}} | {{…}} | {{[…, …]}} | {{…}} | {{…}} | {{H1}} | {{…}} | | |

### L.5 Expected trends and orderings

| Trend | Expected behaviour | Source | Observed (filled after runs) | Status |
|---|---|---|---|---|
| Overall ranking | {{ACRONYM > A > B > …}} | {{…}} | | |
| Scaling with number of agents | {{…}} | {{…}} | | |
| Ablation ordering (largest drop first) | {{w/o C2 > w/o C1 > w/o C3}} | {{…}} | | |
| Learning-curve shape / sample efficiency | {{…}} | {{…}} | | |
| Trade-offs (performance vs. compute, etc.) | {{…}} | {{…}} | | |

### L.6 Link to the claims

| Claim | Values that would support it | Values that would refute it | Observed (filled after runs) | Verdict |
|---|---|---|---|---|
| Cl1 | {{…}} | {{…}} | | |
| Cl2 | {{…}} | {{…}} | | |

### L.7 Comparison after runs (filled once all results are available)

- Share of observed values inside the realistic intervals: {{…%}}
- Rank correlation between expected and observed orderings (Kendall's τ): {{…}}
- Mean absolute deviation (realistic prediction vs. observed): {{…}}

| Unexpected deviation | Investigation performed (bug, config, protocol, version…) | Conclusion | Consequence for the paper |
|---|---|---|---|
| {{…}} | {{…}} | {{artefact fixed / genuine finding}} | {{…}} |

### L.8 Addenda (dated, after freeze)

| Date | Change | Justification | Made before or after seeing results? |
|---|---|---|---|
| {{…}} | {{…}} | {{…}} | {{…}} |
