"""Geometry ported from the frozen 2026-09-18 recognizer; see provenance.json."""
import itertools
import math
import time
import cv2
import numpy as np
from .profiling import timed
from .assignment import AssignmentWorkspace
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class SearchBudgetExceeded(TimeoutError):
    """An incomplete search must not establish a unique correspondence."""


class FrameGeometry:
    """Exact caches live for one input image, including failed H solves."""

    def __init__(self, deadline=None):
        self.deadline = deadline
        self.homographies = {}
        self.observed_hulls = {}
        self.assignment = AssignmentWorkspace()
        self.stats = dict(h_requests=0, h_solves=0, h_hits=0, subsets=0,
                          seed_ms=0., assignment_ms=0., fit_ms=0., group_ms=0.,
                          pair_count=0, edge_count=0, batch_count=0,
                          hull_solves=0, hull_hits=0, graph_pruned_candidates=0,
                          assignment_backend=self.assignment.backend,
                          assignment_native_calls=0, assignment_numpy_calls=0,
                          assignment_fallback_reason=self.assignment.reason)

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchBudgetExceeded("geometry budget exhausted")

    def observed_hull(self, observed):
        key = (observed.shape, observed.dtype.str, observed.tobytes())
        if key not in self.observed_hulls:
            ih = cv2.convexHull(observed.astype(np.float32), returnPoints=False).ravel()
            span = np.linalg.norm(np.ptp(observed, axis=0))
            self.observed_hulls[key] = (ih,
                cv2.contourArea(observed[ih].astype(np.float32)) >= .02*span**2)
            self.stats['hull_solves'] += 1
        else:
            self.stats['hull_hits'] += 1
        return self.observed_hulls[key]

    @timed('homography')
    def homography(self, src, dst, seed_ids=None, stage='fit'):
        self.check()
        started = time.monotonic()
        key = (src.shape, src.dtype.str, src.tobytes(),
               dst.shape, dst.dtype.str, dst.tobytes(),
               None if seed_ids is None else tuple(seed_ids))
        self.stats['h_requests'] += 1
        if key not in self.homographies:
            self.stats['h_solves'] += 1
            self.homographies[key] = (cv2.findHomography(src, dst, 0)[0]
                if seed_ids is None else cv2.getPerspectiveTransform(
                    src[seed_ids].astype(np.float32), dst[seed_ids].astype(np.float32)))
        else:
            self.stats['h_hits'] += 1
        self.stats[stage+'_ms'] += (
            time.monotonic()-started)*1000
        return self.homographies[key]


class TemplateGeometry:
    def __init__(self, template, seed_four=False):
        self.hull = cv2.convexHull(template.astype(np.float32), returnPoints=False).ravel()
        self.source = template[self.hull]
        self.seed_ids = (np.array(max(itertools.combinations(range(len(self.hull)), 4),
            key=lambda ids: cv2.contourArea(self.source[list(ids)].astype(np.float32))))
            if seed_four and len(self.hull) >= 4 else None)
        self.offsets = tuple((reverse, offset) for reverse in (False, True)
                             for offset in range(len(self.hull)))


class CandidateGraph:
    """Stream exact three-radius edges; never allocate a K by K matrix."""

    def __init__(self, candidates, context, expected_counts=None):
        self.candidates, self.context = candidates, context
        self.minimum = min(expected_counts) if expected_counts else 1
        self.components = None

    @timed('candidate_grouping')
    def groups(self, expected):
        count = len(self.candidates)
        if count <= expected + 3:
            return [list(range(count))]
        if self.components is None:
            started = time.monotonic()
            try:
                points = np.array([c.center for c in self.candidates])
                area = np.array([c.core_area for c in self.candidates])
                if not np.isfinite(points).all() or np.any(area <= 0):
                    raise ValueError('finite candidate coordinates and positive areas required')
                tree = cKDTree(points)
                edge_rows, edge_cols, distances, limits = [], [], [], []
                eps = np.finfo(points.dtype).eps
                scale = np.abs(points).max(initial=0.)
                for i in range(count):
                    self.context.check()
                    radius = 32*np.sqrt(area[i])
                    # Cover rounding in the original dtype; exact norm decides every edge.
                    neighbors = np.array(sorted(j for j in tree.query_ball_point(
                        points[i], radius+8*eps*(scale+radius+1)) if j > i), dtype=int)
                    self.context.stats['pair_count'] += len(neighbors)
                    for offset in range(0, len(neighbors), 64):
                        self.context.check()
                        ids = neighbors[offset:offset+64]
                        smaller = np.minimum(area[i], area[ids])
                        distance = np.linalg.norm(points[i]-points[ids], axis=1)
                        comparable = np.maximum(area[i], area[ids]) <= 4*smaller
                        keep = (distance <= 32*np.sqrt(smaller)) & comparable
                        if keep.any():
                            edge_rows.append(np.full(np.count_nonzero(keep), i, dtype=np.int32))
                            edge_cols.append(ids[keep])
                            distances.append(distance[keep])
                            limits.append(np.sqrt(smaller[keep]))
                components = set()
                rows = np.concatenate(edge_rows) if edge_rows else np.array([], dtype=int)
                cols = np.concatenate(edge_cols) if edge_cols else np.array([], dtype=int)
                distances = np.concatenate(distances) if distances else np.array([])
                limits = np.concatenate(limits) if limits else np.array([])
                eligible = np.ones(count, dtype=bool)
                for radius in (32, 8, 16):
                    self.context.check()
                    keep = eligible[rows] & eligible[cols] & (distances <= radius*limits)
                    self.context.stats['edge_count'] += np.count_nonzero(keep)
                    ids = np.flatnonzero(eligible)
                    local = np.full(count, -1, dtype=int)
                    local[ids] = np.arange(len(ids))
                    graph = coo_matrix((np.ones(np.count_nonzero(keep), dtype=np.uint8),
                                        (local[rows[keep]], local[cols[keep]])), shape=(len(ids), len(ids)))
                    _, labels = connected_components(graph, directed=False)
                    grouped = {}
                    for i, label in zip(ids, labels):
                        grouped.setdefault(label, []).append(int(i))
                    components.update(tuple(group) for group in grouped.values()
                                      if len(group) >= self.minimum)
                    if radius == 32:
                        # A smaller-radius component cannot escape its maximum component.
                        eligible[:] = False
                        for group in grouped.values():
                            if len(group) >= self.minimum:
                                eligible[group] = True
                        self.context.stats['graph_pruned_candidates'] += np.count_nonzero(~eligible)
                self.context.check()
                self.components = sorted(components)
            finally:
                self.context.stats['group_ms'] += (time.monotonic()-started)*1000
        return [list(group) for group in self.components if expected <= len(group) <= expected+3]


def _seed_assignments(template, observed, ih, prepared, context, seed_four, tolerance):
    destinations = [observed[np.roll(ih[::-1] if reverse else ih, offset)]
                    for reverse, offset in prepared.offsets]
    for start in range(0, len(destinations), 8):
        context.check()
        predictions = []
        for dst in destinations[start:start+8]:
            H = context.homography(prepared.source, dst,
                                   prepared.seed_ids if seed_four else None, stage='seed')
            predictions.append(None if H is None or not np.isfinite(H).all() else
                cv2.perspectiveTransform(template[None].astype(float), H)[0])
        valid = [p for p in predictions if p is not None]
        if not valid:
            continue
        stamp = time.monotonic()
        assignments, distances = context.assignment.assign(valid, observed)
        context.stats['assignment_ms'] += (time.monotonic()-stamp)*1000
        context.stats['batch_count'] += 1
        context.stats.update(assignment_backend=context.assignment.backend,
            assignment_native_calls=context.assignment.native_calls,
            assignment_numpy_calls=context.assignment.numpy_calls,
            assignment_fallback_reason=context.assignment.reason)
        index = 0
        for predicted in predictions:
            if predicted is None:
                continue
            assignment, nearest = assignments[index], distances[index]
            index += 1
            if len(set(assignment)) == len(template) and nearest.max() <= tolerance*2:
                yield assignment

@timed('pattern_geometry')
def match_pattern(template, points, tolerance=3.0, inverse_camera=None, pixel_scale=1., unordered=False,
                  areas=None, all_orders=False, geometry_ratio=.015,
                  context=None, prepared=None, seed_four=False, validate=None):
    """Enumerate cyclic hull correspondences, then check every selected LED."""
    n = len(template)
    prepared = prepared or TemplateGeometry(template, seed_four)
    hull = prepared.hull
    if len(hull) < 4:
        return '图案需至少四个凸包顶点', [] if all_orders else None
    if len(points) < n:
        return '未匹配：光点不足 / 灯灭 / 遮挡', [] if all_orders else None
    # ponytail: bounded exhaustive subsets; oversized clutter groups are rejected.
    if len(points) > n + 3 or math.comb(len(points), n) > 256:
        return '亮点过多，请提高阈值或减少背景亮点', [] if all_orders else None
    matches = {}
    for combo in itertools.combinations(range(len(points)), n):
        if context is not None:
            context.check()
            context.stats['subsets'] += 1
        if areas is not None:
            selected = np.asarray(areas)[list(combo)]
            # ponytail: equal blocks at equal drive should have comparable cores;
            # reject >2x variation; a radiometric model is needed for unequal drive.
            if selected.max() > 2 * selected.min():
                continue
        observed = points[list(combo)]
        if context is None:
            ih = cv2.convexHull(observed.astype(np.float32), returnPoints=False).ravel()
            if len(ih) != len(hull):
                continue
            span = np.linalg.norm(np.ptp(observed, axis=0))
            usable = not cv2.contourArea(observed[ih].astype(np.float32)) < .02*span**2
        else:
            ih, usable = context.observed_hull(observed)
        if len(ih) != len(hull) or not usable:
            continue
        def original_assignments():
            for order in (ih, ih[::-1]):
                for offset in range(len(hull)):
                    dst = observed[np.roll(order, offset)]
                    H, _ = cv2.findHomography(template[hull], dst, 0)
                    if H is None or not np.isfinite(H).all():
                        continue
                    predicted = cv2.perspectiveTransform(template[None].astype(float), H)[0]
                    distances = np.linalg.norm(predicted[:, None] - observed[None], axis=2)
                    assignment = distances.argmin(axis=1)
                    if len(set(assignment)) != n or distances[np.arange(n), assignment].max() > tolerance * 2:
                        continue
                    yield assignment
        assignments = (original_assignments() if context is None else
            _seed_assignments(template, observed, ih, prepared, context, seed_four, tolerance))
        for assignment in assignments:
            ordered = observed[assignment]
            H = (cv2.findHomography(template, ordered, 0)[0] if context is None else
                 context.homography(template, ordered))
            if H is None or not np.isfinite(H).all():
                continue
            if inverse_camera is not None:
                # A calibrated rigid plane has equally scaled, orthogonal axes.
                # Loose bounds allow pixel noise on tiny targets, but reject
                # arbitrary projective fits to keyboard/monitor reflections.
                axes = (inverse_camera @ H)[:, :2]
                lengths = np.linalg.norm(axes, axis=0)
                if (lengths.min() < 1e-12 or lengths.max() > 2 * lengths.min()
                        or abs(axes[:, 0] @ axes[:, 1]) > .5 * np.prod(lengths)):
                    continue
            # A single visible plane must not cross the projective horizon.
            denominator = np.column_stack((template, np.ones(n))) @ H[2]
            if not (np.all(denominator > 1e-8) or np.all(denominator < -1e-8)):
                continue
            fitted = cv2.perspectiveTransform(template[None].astype(float), H)[0]
            errors = np.linalg.norm(fitted - ordered, axis=1)
            span = np.linalg.norm(np.ptp(ordered, axis=0))
            limit = min(tolerance, span * geometry_ratio)
            separation = np.linalg.norm(ordered[:, None] - ordered[None], axis=2)
            np.fill_diagonal(separation, np.inf)
            # Resolve each pair rather than rejecting an otherwise accurate
            # small target solely because its bounding span is under 24 px.
            if separation.min() < 2 * pixel_scale or errors.max() > limit:
                continue
            if validate is not None and not validate(ordered):
                continue
            ids = tuple(combo[i] for i in assignment)
            matches[ids] = float(np.sqrt(np.mean(errors ** 2)))
    if all_orders:
        return '候选对应（不保证唯一）', [np.array(order) for order in sorted(matches, key=matches.get)]
    if unordered:
        # Same observed set under several symmetric labelings is one detected
        # layout. Different point sets must still compete and remain rejectable.
        sets = {}
        for order, error in matches.items():
            key = tuple(sorted(order))
            sets[key] = min(sets.get(key, float('inf')), error)
        matches = sets
    ranked = sorted(matches, key=matches.get)
    if not ranked:
        return '未匹配：几何关系不符', None
    if len(ranked) > 1 and matches[ranked[1]] - matches[ranked[0]] < .5 * pixel_scale:
        return '匹配歧义：图案对称或有多个相似目标', None
    return f'匹配当前图案 {n}/{n} · 误差 {matches[ranked[0]]:.2f} px', np.array(ranked[0])

def candidate_groups(candidates, expected):
    """Separate nearby spots of comparable area before geometric subset matching."""
    # ponytail: cap clutter before the O(n^2) proximity graph; raise only after
    # profiling real frames and replacing this with a spatial index.
    if len(candidates) > 256:
        return []
    if len(candidates) <= expected + 3:
        return [list(range(len(candidates)))]
    points = np.array([c.center for c in candidates])
    area = np.array([c.core_area for c in candidates])
    smaller = np.minimum(area[:, None], area[None])
    distance = np.linalg.norm(points[:, None] - points[None], axis=2)
    comparable = np.maximum(area[:, None], area[None]) <= 4 * smaller
    groups = set()
    # Multiple radii cover compact saturated spots and sparse, sharply focused LEDs.
    for radius in (8, 16, 32):
        adjacent = (distance <= radius * np.sqrt(smaller)) & comparable
        seen = set()
        for seed in range(len(candidates)):
            if seed in seen:
                continue
            group, pending = {seed}, [seed]
            seen.add(seed)
            while pending:
                neighbors = set(np.flatnonzero(adjacent[pending.pop()])) - seen
                group.update(neighbors)
                seen.update(neighbors)
                pending.extend(neighbors)
            if expected <= len(group) <= expected + 3:
                groups.add(tuple(sorted(group)))
    return [list(group) for group in sorted(groups)]
