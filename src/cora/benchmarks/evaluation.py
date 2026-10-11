"""Explicit evaluation definitions; no learner-facing oracle access."""
import numpy as np


def recovery_time(performance, disruption_step, reference, fraction=.9, consecutive=2):
    """First sustained return to a positive pre-disruption reference, or None."""
    values=np.asarray(performance,float)
    if not 0<=disruption_step<len(values) or reference<=0 or not 0<fraction<=1 or consecutive<1 or not np.isfinite(values).all():
        raise ValueError('Valid positive-reference recovery protocol required')
    for t in range(disruption_step,len(values)-consecutive+1):
        if np.all(values[t:t+consecutive]>=fraction*reference):
            return t-disruption_step
    return None


def performance_profile(regrets, thresholds):
    """Fraction of seed/problem regret observations below each fixed threshold."""
    x=np.asarray(regrets,float); thresholds=np.asarray(thresholds,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or thresholds.ndim!=1 or not np.isfinite(thresholds).all():
        raise ValueError('Finite vectors required')
    return np.mean(x[:,None]<=thresholds[None,:],axis=0)


def structure_accuracy(truth_terms,inferred_terms):
    truth=set(map(tuple,truth_terms))-{()}; inferred=set(map(tuple,inferred_terms))-{()}
    overlap=len(truth&inferred)
    return {'precision':overlap/len(inferred) if inferred else None,
            'recall':overlap/len(truth) if truth else None,
            'false_omissions':len(truth-inferred),'false_inclusions':len(inferred-truth)}


def simultaneous_coverage(truth,prediction,radius):
    truth,prediction,radius=map(lambda x:np.asarray(x,float),(truth,prediction,radius))
    if truth.shape!=prediction.shape or radius.shape!=truth.shape or not all(np.isfinite(a).all() for a in (truth,prediction,radius)) or np.any(radius<0):
        raise ValueError('Finite matching arrays and nonnegative radii required')
    return bool(np.all(abs(truth-prediction)<=radius))
