"""Prediction-guided current LED association; all geometry/pose gates remain."""
import cv2
import numpy as np

from .geometry import match_pattern
from .profiling import timed


@timed('fast_association')
def associate(predicted, candidates):
    points = np.asarray([c.center for c in candidates], dtype=float).reshape(-1, 2)
    predicted = np.asarray(predicted, dtype=float)
    if (len(points) < len(predicted) or len(predicted) < 4 or
            not np.isfinite(points).all() or not np.isfinite(predicted).all()):
        return None, 'insufficient_or_invalid_points'
    distance = np.linalg.norm(predicted[:, None]-points[None], axis=2)
    neighbors = np.argsort(distance, axis=1, kind='stable')[:, :2]
    selected = neighbors[:, 0]
    if len(set(selected)) != len(predicted):
        return None, 'repeated_assignment'
    separation = np.linalg.norm(predicted[:, None]-predicted[None], axis=2)
    np.fill_diagonal(separation, np.inf)
    radius = min(5., .35*float(separation.min()))
    nearest = distance[np.arange(len(predicted)), selected]
    if np.any(nearest > radius):
        return None, 'distance_gate'
    second = distance[np.arange(len(predicted)), neighbors[:, 1]]
    if np.any(second-nearest < .75):
        return None, 'neighbor_ambiguity'
    area = np.asarray([candidates[i].core_area for i in selected])
    if np.any(area <= 0) or area.max() > 4*area.min():
        return None, 'area_gate'
    return selected, 'associated'


@timed('fast_association_validation')
def validate(tracker, pattern, raw, recovery, context):
    corrected = cv2.fisheye.undistortPoints(raw[:, None], tracker.K, tracker.D, P=tracker.K)[:, 0]
    if not np.isfinite(corrected).all() or np.abs(corrected).max() > 1e6:
        return False
    # Check alternate labelings of the selected set with the existing ambiguity gate.
    status, order = match_pattern(pattern['template'], corrected,
        tolerance=3. if recovery else 6., inverse_camera=None if recovery else tracker.inverse,
        pixel_scale=1. if recovery else 2., geometry_ratio=.025 if recovery else .015,
        context=context, prepared=tracker.prepared[pattern['id']], seed_four=not recovery)
    return order is not None and np.array_equal(order, np.arange(len(raw)))
