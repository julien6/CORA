"""Pessimistic diagnostic comparison with optimistic no-op fallback."""
import numpy as np

from cora.interventions import Intervention, candidates


def select(spec, org, options, features, model, swap_cost, structural_radius=None, shift_radius=None):
    if not options or Intervention() not in options:
        raise ValueError("No-op must be a candidate")
    valid = set(candidates(spec, org))
    if len(set(options)) != len(options) or any(z not in valid for z in options):
        raise ValueError("Every candidate must be unique and admissible")
    if np.asarray(features).shape != (len(options), len(model.coefficients)):
        raise ValueError("Candidate feature dimensions do not match")
    mean = np.asarray(model.predict(features))
    radius = np.asarray(model.radius(features)).copy()
    for extra in (structural_radius, shift_radius):
        if extra is not None:
            extra = np.asarray(extra)
            if extra.shape != radius.shape or np.any(extra < 0) or not np.isfinite(extra).all():
                raise ValueError("Invalid approximation radius")
            radius += extra
    net = mean - np.asarray([z.cost(swap_cost) for z in options])
    if not np.isfinite(net).all() or not np.isfinite(radius).all():
        raise ValueError("Non-finite prediction or diagnostic radius")
    noop = options.index(Intervention())
    lower = net - radius
    best = int(np.argmax(lower))
    identified = np.asarray(model.identifiable(features))
    # Ridge does not cure identifiability; abstain on unsupported targets.
    accepted = best != noop and identified[best] and identified[noop] and lower[best] > net[noop] + radius[noop]
    chosen = best if accepted else noop
    return options[chosen], {
        "candidate_bits": [list(z.bits) for z in options],
        "predicted_value": mean.tolist(), "predicted_net_value": net.tolist(),
        "diagnostic_radius": radius.tolist(), "value_identifiable": identified.tolist(),
        "selected_index": chosen, "certified": False,
        "structural_bound_verified": False, "shift_bound_verified": False,
    }
