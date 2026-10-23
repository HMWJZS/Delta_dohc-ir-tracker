"""RPG-inspired causal 2D LED prediction; predictions are never observations."""
import numpy as np

from .config import CONFIG
from .profiling import timed


class ROIMotion:
    def __init__(self):
        self.history = {}
        self.coverage_history = {}

    def discard(self, name, keep_coverage=False):
        self.history.pop(name, None)
        if not keep_coverage:
            self.coverage_history.pop(name, None)

    def commit(self, name, points, led_ids, stamp, completed, confirmed):
        points = np.asarray(points, dtype=float)
        if (not np.isfinite(points).all() or points.shape != (len(led_ids), 2) or not len(led_ids)
                or not np.isfinite(stamp) or not np.isfinite(completed)
                or not 0 <= completed-stamp <= CONFIG['measurement_max_age_seconds']):
            self.discard(name)
            return
        backup = self.coverage_history.get(name, [])
        if backup and (backup[-1][2] != tuple(led_ids) or
                       not 0 < stamp-backup[-1][0] <= CONFIG['roi_prediction_max_age_seconds']):
            self.coverage_history.pop(name, None)
        if not confirmed:
            self.discard(name, keep_coverage=True)
            return
        backup = self.coverage_history.get(name, [])
        self.coverage_history[name] = (backup+[(stamp, points.copy(), tuple(led_ids))])[-2:]
        old = self.history.get(name, [])
        if old and (old[-1][2] != tuple(led_ids) or
                    not 0 < stamp-old[-1][0] <= CONFIG['roi_prediction_max_age_seconds']):
            old = []
        self.history[name] = (old+[(stamp, points.copy(), tuple(led_ids))])[-2:]

    @timed('roi_prediction')
    def predict(self, name, stamp, size):
        old = self.history.get(name, [])
        audit = dict(enabled=CONFIG['roi_prediction_enabled'], used=False, reason='insufficient_history')
        if not audit['enabled']:
            audit['reason'] = 'disabled'
            return None, audit
        if len(old) < 2:
            return None, audit
        (ta, a, ids), (tb, b, _) = old
        audit['history_source_monotonic'] = [ta, tb]
        age = CONFIG['roi_prediction_max_age_seconds']
        if not 0 < stamp-tb <= age or not 0 < tb-ta <= age:
            self.discard(name)
            audit['reason'] = 'history_expired_or_invalid_time'
            return None, audit
        ratio = (stamp-tb)/(tb-ta)
        audit['extrapolation_ratio'] = ratio
        if ratio > CONFIG['roi_prediction_max_ratio']:
            audit['reason'] = 'extrapolation_limit'
            return None, audit
        predicted = b+(b-a)*ratio
        if not np.isfinite(predicted).all() or np.any(predicted < 0) or np.any(predicted >= size):
            audit['reason'] = 'prediction_outside_image'
            return None, audit
        audit.update(used=True, reason='constant_velocity', max_displacement_px=float(
            np.linalg.norm(predicted-b, axis=1).max()))
        return predicted, audit

    def rectangle(self, name, points, old_stamp, stamp, size, expanded=False):
        predicted, audit = self.predict(name, stamp, size)
        margin = max(12., .5*np.linalg.norm(np.ptp(points, axis=0)))+120*(stamp-old_stamp)
        if predicted is not None:
            margin = max(margin, 2*audit['max_displacement_px']+8.)
        def bounds(p, m):
            lo = np.maximum(np.floor(p.min(0)-m), 0).astype(int)
            hi = np.minimum(np.ceil(p.max(0)+m+1), size).astype(int)
            return [int(v) for v in (*lo, *hi)]
        rectangle = bounds(points if predicted is None else predicted, margin*(2 if expanded else 1))
        if predicted is not None:
            x0, y0, x1, y1 = rectangle
            if (x1-x0)*(y1-y0) > CONFIG['roi_prediction_max_area_fraction']*np.prod(size):
                predicted = None
                audit.update(used=False, reason='prediction_area_limit')
                margin = max(12., .5*np.linalg.norm(np.ptp(points, axis=0)))+120*(stamp-old_stamp)
                rectangle = bounds(points, margin*(2 if expanded else 1))
        audit['roi'] = rectangle
        return rectangle, predicted, audit

    @timed('roi_coverage')
    def cover_rectangles(self, rectangles, prediction, stamp, size):
        """Bound search coverage only; never create points for association or pose."""
        from .methods import merge_rectangles
        enabled = CONFIG['roi_coverage_enabled']
        covered, audits = dict(rectangles), {}
        def area(r):
            return (r[2]-r[0])*(r[3]-r[1])
        for name, original in rectangles.items():
            audit = dict(enabled=enabled, applied=False, reason='disabled' if not enabled else 'prediction_available',
                         original_roi=original, roi=original, area_ratio=1.)
            audits[name] = audit
            if not enabled:
                continue
            old = self.coverage_history.get(name, [])
            if old:
                audit['history_source_monotonic'] = [p[0] for p in old]
                audit['age_seconds'] = stamp-old[-1][0]
                if not 0 < stamp-old[-1][0] <= CONFIG['roi_prediction_max_age_seconds']:
                    self.coverage_history.pop(name, None)
                    audit['reason'] = 'history_expired_or_invalid_time'
                    continue
            if prediction.get(name, {}).get('reason') not in ('extrapolation_limit', 'insufficient_history'):
                audit['reason'] = 'strict_prediction_not_eligible'
                continue
            if len(old) != 2:
                audit['reason'] = 'insufficient_coverage_history'
                continue
            (ta, a, ids), (tb, b, other_ids) = old
            if ids != other_ids or not 0 < tb-ta <= CONFIG['roi_prediction_max_age_seconds']:
                self.coverage_history.pop(name, None)
                audit['reason'] = 'invalid_coverage_history'
                continue
            age = stamp-tb
            moved = b+(b-a)*(age/(tb-ta))
            margin = max(12., .5*np.linalg.norm(np.ptp(b, axis=0)))+120*age
            points = np.concatenate((b, moved))
            lo = np.maximum(np.minimum(np.floor(points.min(0)-margin), original[:2]), 0)
            hi = np.minimum(np.maximum(np.ceil(points.max(0)+margin+1), original[2:]), size)
            if not np.isfinite(points).all():
                audit['reason'] = 'invalid_coverage_points'
                continue
            candidate = [int(v) for v in (*lo, *hi)]
            ratio = area(candidate)/max(1, area(original))
            audit['candidate_roi'], audit['candidate_area_ratio'] = candidate, ratio
            if (ratio > CONFIG['roi_coverage_max_area_ratio'] or
                    area(candidate) > CONFIG['roi_coverage_max_area_fraction']*np.prod(size)):
                audit['reason'] = 'coverage_area_limit'
                continue
            covered[name] = candidate
            audit.update(applied=candidate != original, reason='bounded_motion_coverage', roi=candidate, area_ratio=ratio)
        if any(a['applied'] for a in audits.values()):
            original_area = sum(map(area, merge_rectangles(list(rectangles.values()))))
            covered_area = sum(map(area, merge_rectangles(list(covered.values()))))
            if covered_area > CONFIG['roi_coverage_max_area_ratio']*original_area:
                covered = dict(rectangles)
                for name, audit in audits.items():
                    if audit['applied']:
                        audit.update(applied=False, reason='merged_area_limit', roi=rectangles[name], area_ratio=1.)
        return covered, audits
