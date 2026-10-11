"""Finite organizational MDP planning with explicit transition/reward model.

Generic cost-aware planning comparator; not a reproduction of an unnamed paper.
"""
import numpy as np


def value_iteration(transitions,rewards,costs,valid,gamma=.95,tolerance=1e-8,max_iterations=10000):
    p,r,c=np.asarray(transitions,float),np.asarray(rewards,float),np.asarray(costs,float)
    valid=np.asarray(valid,bool)
    if p.ndim!=3 or p.shape[0]!=p.shape[2] or r.shape!=p.shape[:2] or c.shape!=r.shape or valid.shape!=r.shape or not np.all(valid.any(axis=1)):
        raise ValueError('Dimensions must be (state, action, next_state)')
    if not all(np.isfinite(a).all() for a in (p,r,c)) or np.any(p<0) or np.any(c<0) or not np.allclose(p.sum(axis=2),1) or not 0<=gamma<1 or not np.isfinite(tolerance) or tolerance<=0 or max_iterations<1:
        raise ValueError('Stochastic transitions and finite discounted MDP required')
    v=np.zeros(len(p))
    for iteration in range(max_iterations):
        q=np.where(valid,r-c+gamma*np.einsum('sak,k->sa',p,v),-np.inf)
        updated=q.max(axis=1)
        residual=float(np.max(abs(updated-v)))
        v=updated
        if residual<=tolerance:break
    q=np.where(valid,r-c+gamma*np.einsum('sak,k->sa',p,v),-np.inf)
    return {'values':v,'policy':q.argmax(axis=1),'iterations':iteration+1,'converged':residual<=tolerance,'value_error_upper_bound':residual/(1-gamma)}
