"""Host-side typed organizational edits with centralized atomic commit."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SymbolicOrganization:
    agents: tuple[str,...]
    roles: tuple[str,...]
    missions: tuple[str,...]
    assignments: tuple[int,...]
    agent_roles: tuple[int,...]
    permissions: frozenset[tuple[int,int]]
    prohibitions: frozenset[tuple[int,int]] = frozenset()
    obligations: frozenset[tuple[int,int]] = frozenset()
    relations: frozenset[tuple[str,int,int]] = frozenset()
    prerequisites: frozenset[tuple[int,int]] = frozenset()
    version: int = 0

    def validate(self):
        n,r,m=len(self.agents),len(self.roles),len(self.missions)
        if min(n,r,m)<1 or any(len(set(x))!=len(x) for x in (self.agents,self.roles,self.missions)):
            raise ValueError('Unique nonempty organization entities required')
        if len(self.assignments)!=n or len(self.agent_roles)!=n or self.version<0:
            raise ValueError('Malformed organization')
        if any(not 0<=a<m for a in self.assignments) or any(not 0<=a<r for a in self.agent_roles):
            raise ValueError('Unknown role or mission')
        if set(self.assignments)!=set(range(m)):
            raise ValueError('Every mission requires coverage')
        for role,mission in self.permissions|self.prohibitions|self.obligations:
            if not 0<=role<r or not 0<=mission<m:
                raise ValueError('Unknown normative entity')
        assigned=set(zip(self.agent_roles,self.assignments))
        if not assigned<=self.permissions or assigned&self.prohibitions or not self.obligations<=assigned:
            raise ValueError('Permission, prohibition or obligation violated')
        for kind,a,b in self.relations:
            if kind not in ('delegates','communicates','supervises') or not 0<=a<n or not 0<=b<n or a==b:
                raise ValueError('Invalid typed organizational relation')
        graph={i:set() for i in range(m)}
        for a,b in self.prerequisites:
            if not 0<=a<m or not 0<=b<m or a==b:
                raise ValueError('Invalid prerequisite')
            graph[b].add(a)
        active,visited=set(),set()
        def visit(i):
            if i in active:
                raise ValueError('Cyclic mission prerequisites')
            if i in visited:
                return
            active.add(i)
            for j in graph[i]:visit(j)
            active.remove(i);visited.add(i)
        for i in graph:visit(i)


@dataclass(frozen=True)
class Edit:
    kind: str
    first: int
    second: int
    relation: str = 'delegates'


def transaction(org,edits,expected_version):
    """Ordered edit list; validate only final state; stale version rejects commit."""
    org.validate()
    if expected_version!=org.version:
        raise ValueError('Stale organizational proposal')
    result=org
    for edit in edits:
        if edit.kind in ('assign','role'):
            values=list(result.assignments if edit.kind=='assign' else result.agent_roles)
            if not 0<=edit.first<len(values):raise ValueError('Unknown agent')
            values[edit.first]=edit.second
            result=replace(result,**{('assignments' if edit.kind=='assign' else 'agent_roles'):tuple(values)})
        elif edit.kind in ('grant','revoke','prohibit','permit','oblige','discharge'):
            field={'grant':'permissions','revoke':'permissions','prohibit':'prohibitions','permit':'prohibitions','oblige':'obligations','discharge':'obligations'}[edit.kind]
            values=set(getattr(result,field)); pair=(edit.first,edit.second)
            if edit.kind in ('grant','prohibit','oblige'):values.add(pair)
            else:values.discard(pair)
            result=replace(result,**{field:frozenset(values)})
        elif edit.kind in ('link','unlink'):
            values=set(result.relations); relation=(edit.relation,edit.first,edit.second)
            if edit.kind=='link':values.add(relation)
            else:values.discard(relation)
            result=replace(result,relations=frozenset(values))
        else:raise ValueError('Unknown organizational edit')
    result.validate()
    return replace(result,version=org.version+1) if edits else org
