"""Algorithm 1 orchestration with explicit policy-version estimand boundaries.

Trainer and trial factories are injected; no expensive defaults or hidden runs.
Organizational adoption is centralized and atomic through caller's commit.
"""
from dataclasses import dataclass
from cora.benchmarks.methods import run_method
from cora.temporal import PolicyVersionedData


@dataclass(frozen=True)
class AdaptationBudget:
    rounds: int = 2
    training_steps_per_round: int = 20
    intervention_queries_per_round: int = 4

    def __post_init__(self):
        if min(self.rounds,self.training_steps_per_round,self.intervention_queries_per_round)<1:
            raise ValueError('Positive explicit budgets required')


def adapt(train_and_freeze, make_trial, commit, budget, seed=0, estimation=None):
    """Injected contracts:

    train_and_freeze(round, steps) -> (policy, unique_version, actual_steps).
    make_trial(policy, version, round) -> public, observe, pre-treatment context.
    commit(edit_bits, expected_organization_version) -> next organization version.

    A rejected/conflicting commit must raise; only successful adoption is logged.
    Public metadata is the only learner input. Oracle estimates are not accessed.
    """
    estimation=estimation or {'ridge':.01,'diagnostic_scale':1.}
    events=[]; versions=set(); organization_version=0
    for round_index in range(budget.rounds):
        policy,version,actual_steps=train_and_freeze(round_index,budget.training_steps_per_round)
        if version in versions or actual_steps!=budget.training_steps_per_round:
            raise ValueError('Unique frozen policy version and exact training budget required')
        versions.add(version)
        public,observe,context=make_trial(policy,version,round_index)
        if public.policy_version!=version:
            raise ValueError('Trial continuation does not match frozen policy')
        data=PolicyVersionedData(version,context,[])
        decisions,logs=run_method('cora',public,observe,[budget.intervention_queries_per_round],seed+round_index,estimation)
        for record in logs:
            data.append(record['policy_version'],context,record)
        chosen=decisions[-1]['selected_index']
        if not public.valid[chosen]:
            raise RuntimeError('Invalid organizational decision')
        if chosen!=public.noop:
            next_version=commit(public.bits[chosen].copy(),organization_version)
            if not isinstance(next_version, int) or next_version <= organization_version:
                raise RuntimeError('Successful commit must advance organization version')
            organization_version=next_version
        events.append({'round':round_index,'training_steps':actual_steps,'policy_version':version,'context':context,'organization_version':organization_version,'experiments':data.records,'decision':decisions[-1],'continuation':'frozen_per_round','carried_effect_estimates':False})
    return events
