"""Explicit acyclic temporal ancestry, separate from symbolic organizations."""
from dataclasses import dataclass
from cora.features import FeatureEncoder


@dataclass(frozen=True)
class TemporalGraph:
    parents: dict[str, tuple[str, ...]]
    interventions: tuple[str, ...]
    returns: tuple[str, ...]

    def ancestors(self, node):
        if node not in self.parents:
            raise ValueError(f"Unknown causal node: {node}")
        visited, active = set(), set()
        def visit(n):
            if n in active:
                raise ValueError("Temporal SCM must be acyclic")
            if n in visited:
                return
            if n not in self.parents:
                raise ValueError(f"Unknown parent: {n}")
            active.add(n)
            for p in self.parents[n]:
                visit(p)
            active.remove(n)
            visited.add(n)
        visit(node)
        return visited - {node}

    def encoder(self):
        if len(set(self.interventions)) != len(self.interventions) or not self.interventions or not self.returns:
            raise ValueError("Unique intervention nodes and return factors required")
        for n in self.parents:
            self.ancestors(n)
        if any(n not in self.parents or self.parents[n] for n in self.interventions):
            raise ValueError("Interventions must be root nodes")
        edges = tuple(tuple(i for i, u in enumerate(self.interventions) if u in self.ancestors(y)) for y in self.returns)
        return FeatureEncoder.from_hyperedges(len(self.interventions), edges)


@dataclass
class PolicyVersionedData:
    """Reference-policy estimates never silently survive a policy/context update."""
    policy_version: str
    context: str
    records: list

    def append(self, policy_version, context, record):
        if policy_version != self.policy_version or context != self.context:
            raise ValueError("New estimand: create a separate dataset")
        self.records.append(record)
