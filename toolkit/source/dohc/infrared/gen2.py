"""Gen2: frozen U+G measurements, bounded native ROI/discovery and exact caches."""
import time
import re
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .detector import Candidate, CPUExtractor, MultiTracker, reject_conflicts
from .geometry import (CandidateGraph, FrameGeometry, SearchBudgetExceeded,
                       TemplateGeometry, match_pattern)
from .version import digest, module_identity
from .pose_resolution import PoseResolver
from .roi_motion import ROIMotion
from .roi_association import associate, validate as validate_association

from .image_source import crop_bgr, full_bgr
from .config import CONFIG
from .profiling import call, timed, profiled, span


def expanded_region(rectangle, shape):
    x0, y0, x1, y1 = rectangle
    height, width = shape[:2]
    halo = CONFIG['roi_preprocess_halo_px']
    return [max(0, x0-halo), max(0, y0-halo), min(width, x1+halo), min(height, y1+halo)]


class RegionBoundary(RuntimeError):
    """Foreground touched an artificial crop boundary; identify its owning ROI."""

    def __init__(self, rectangle):
        super().__init__('foreground at artificial ROI boundary')
        self.rectangle = list(rectangle)


class RegionExtractor(CPUExtractor):
    name = 'cpu_gen2'

    def __init__(self, rectangles, recovery=False, context=None):
        from .methods import merge_rectangles
        self.rectangles = merge_rectangles(rectangles)
        self.recovery, self.context = recovery, context
        self.grids = {}
        self.regions = None
        self.calls = self.moment_calls = 0

    def prepare(self, gray, weights=None):
        # Full-image smoothing preserves original reflection/context even in a ROI.
        super().prepare(gray, weights)
        self.prepare_patches([(r, gray[r[1]:r[3], r[0]:r[2]],
                               self.smooth[r[1]:r[3], r[0]:r[2]]) for r in self.rectangles],
                             gray.shape, int(self.smooth.max()))

    def prepare_patches(self, patches, shape, maximum):
        self.patches, self.shape = patches, shape
        self.regions, self.grids = None, {}
        self.maximum = maximum

    def check_boundary(self, threshold):
        height, width = self.shape
        for (x0, y0, x1, y1), _, core in self.patches:
            edges = []
            if y0: edges.append(core[0])
            if y1 < height: edges.append(core[-1])
            if x0: edges.append(core[:, 0])
            if x1 < width: edges.append(core[:, -1])
            if any(np.any(edge > threshold) for edge in edges):
                raise RegionBoundary([x0, y0, x1, y1])

    @timed('extract_threshold')
    def extract(self, threshold, min_area=2):
        if threshold >= self.maximum:
            return []
        self.check_boundary(threshold)
        if not self.recovery:
            found = []
            for (x0, y0, x1, y1), raw, smooth in self.patches:
                self.context.check()
                self.calls += 1
                count, labels, stats, _ = call('connected_components', cv2.connectedComponentsWithStats,
                    (smooth > threshold).astype(np.uint8), connectivity=8)
                eligible = np.flatnonzero(stats[1:, 4] >= min_area)+1
                if len(eligible) > 2048:
                    # A smooth component only counts if raw weights contain a
                    # positive pixel. Exactly the frozen fast-path rejection.
                    positive = np.bincount(labels[raw >= threshold], minlength=count)
                    if np.count_nonzero(positive[eligible]) > 2048:
                        return [Candidate(np.zeros(2, np.float32), 0, 0)]*2049
                for label in eligible:
                    self.context.check()
                    x, y, w, h, area = map(int, stats[label])
                    with span('centroid_moments'):
                        patch = raw[y:y+h, x:x+w]
                        mask = labels[y:y+h, x:x+w] == label
                        weights = np.where(mask, np.maximum(patch.astype(np.int32)-threshold+1, 0), 0)
                        total = weights.sum(dtype=np.int64)
                        if not total:
                            continue
                        key = ('integer', patch.shape)
                        if key not in self.grids:
                            self.grids[key] = np.indices(patch.shape)
                        yy, xx = self.grids[key]
                        local = np.float32([x+(xx*weights).sum()/total, y+(yy*weights).sum()/total])
                        found.append(Candidate(local+np.float32([x0, y0]), int(patch[mask].max()), area))
                    if len(found) > 2048:
                        return found[:2049]
                if len(found) > 2048:
                    return found[:2049]
            found.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
            return found
        if self.regions is None:
            self.regions = []
            for (ox, oy, x1, y1), region_raw, smooth in self.patches:
                self.context.check()
                count, labels, stats, _ = call('weak_region_components', cv2.connectedComponentsWithStats,
                    (smooth > 9).astype(np.uint8), connectivity=8)
                for label in range(1, count):
                    self.context.check()
                    x, y, w, h, area = map(int, stats[label])
                    if area < 1:
                        continue
                    sm = np.where(labels[y:y+h, x:x+w] == label, smooth[y:y+h, x:x+w], 0)
                    raw = region_raw[y:y+h, x:x+w]
                    self.regions.append((ox+x, oy+y, raw, sm, int(sm.max())))
        found = []
        for ox, oy, raw, smooth, peak in self.regions:
            self.context.check()
            if peak <= threshold:
                continue
            self.calls += 1
            count, labels, stats, _ = call('connected_components', cv2.connectedComponentsWithStats,
                (smooth > threshold).astype(np.uint8), connectivity=8)
            for label in range(1, count):
                self.context.check()
                x, y, w, h, area = map(int, stats[label])
                if area < min_area:
                    continue
                patch = raw[y:y+h, x:x+w].astype(np.float32)
                mask = labels[y:y+h, x:x+w] == label
                weights = np.where(mask, np.maximum(patch-threshold+1., 0.), 0.)
                total = float(weights.sum())
                self.moment_calls += 1
                if total <= 0:
                    continue
                if patch.shape not in self.grids:
                    self.grids[patch.shape] = np.indices(patch.shape, dtype=np.float32)
                yy, xx = self.grids[patch.shape]
                center = np.float32([ox+x+float((xx*weights).sum()/total),
                                     oy+y+float((yy*weights).sum()/total)])
                found.append(Candidate(center, int(patch[mask].max()), area))
                if len(found) > 2048:
                    return found
        found.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
        return found


class GPURegionExtractor:
    """Full-image GPU CCL, or exact CPU CCL on downloaded ROI planes."""
    def __init__(self, gpu, branch, rectangles, context, components):
        self.gpu, self.context = gpu, context
        self.calls = self.moment_calls = 0
        self.maximum = call('local_preprocess', gpu.prepare_gen2, branch)
        self.cpu = None
        full = rectangles == [[0, 0, int(gpu.w), int(gpu.h)]]
        if components == 'cpu' or not full:
            context.check()
            self.cpu = RegionExtractor(rectangles, context=context)
            self.cpu.prepare_patches(call('gpu_download', gpu.download_regions, self.cpu.rectangles), gpu.shape, self.maximum)

    def extract(self, threshold, min_area=2):
        self.context.check()
        if threshold >= self.maximum:
            return []
        if self.cpu is not None:
            result = self.cpu.extract(threshold, min_area)
            self.calls, self.moment_calls = self.cpu.calls, self.cpu.moment_calls
            return result
        self.calls += 1
        return self.gpu.extract(threshold, min_area)


@timed('recovery_centers')
def recovery_centers(candidates, intensity, context, cache=None, image=None):
    """Frozen annulus formula, sparse neighbor query and frame-local grid reuse."""
    if not candidates:
        return candidates
    points = np.array([c.center for c in candidates])
    tree = cKDTree(points)
    eps = np.finfo(points.dtype).eps
    margin = 8*eps*(np.abs(points).max()+21)
    grids, centers = ({}, {}) if cache is None else cache
    rebuilt = []
    for i, candidate in enumerate(candidates):
        context.check()
        neighbors = [j for j in tree.query_ball_point(points[i], 20+margin) if j != i]
        nearest = (np.linalg.norm(points[i]-points[neighbors], axis=1).min()
                   if neighbors else np.inf)
        radius = int(np.clip(np.floor(nearest*.35), 2, 6))
        key = (candidate.center.dtype.str, candidate.center.tobytes(), radius)
        if key not in centers:
            x, y = map(float, candidate.center)
            outer = radius+2
            x0, y0 = max(0, int(np.floor(x-outer))), max(0, int(np.floor(y-outer)))
            shape = intensity.shape if intensity is not None else image.shape
            x1 = min(shape[1], int(np.ceil(x+outer))+1)
            y1 = min(shape[0], int(np.ceil(y+outer))+1)
            patch = (intensity[y0:y1, x0:x1] if intensity is not None else
                     cv2.min(crop_bgr(image, [x0, y0, x1, y1])[:, :, 0],
                             crop_bgr(image, [x0, y0, x1, y1])[:, :, 2])).astype(np.float32)
            if patch.shape not in grids:
                grids[patch.shape] = np.indices(patch.shape, dtype=np.float32)
            yy, xx = grids[patch.shape]
            yy, xx = yy+y0, xx+x0
            d2 = (xx-x)**2+(yy-y)**2
            ring = (d2 > radius**2) & (d2 <= outer**2)
            center = candidate.center
            if ring.any():
                weights = np.maximum(patch-np.median(patch[ring]), 0)*(d2 <= radius**2)
                total = float(weights.sum())
                if total > 0:
                    proposed = np.float32([(weights*xx).sum()/total, (weights*yy).sum()/total])
                    if np.linalg.norm(proposed-candidate.center) <= 1.5:
                        center = proposed
            centers[key] = center
        rebuilt.append(Candidate(centers[key], candidate.peak, candidate.core_area))
    return rebuilt


class Gen2Tracker(MultiTracker):
    MAX_CANDIDATES = 2048

    def __init__(self, calibration, patterns, extractor=None, gpu_components='auto'):
        if getattr(extractor, 'is_npu', False):
            raise ValueError('Gen2 supports CPU or CPU+GPU; no NPU')
        if gpu_components not in ('auto', 'cpu'):
            raise ValueError('Gen2 GPU components must be auto or cpu')
        self.gpu = extractor if getattr(extractor, 'is_gpu', False) else None
        self.gpu_components = gpu_components
        self.backend = 'cpu_gpu_gen2' if self.gpu else 'cpu_gen2'
        if not isinstance(patterns, list):
            raise ValueError('Tracker registry must be a list')
        selected = {}
        for entry in patterns:
            if not isinstance(entry, dict) or not isinstance(entry.get('id'), str):
                raise ValueError('Tracker patterns require string IDs')
            slots = re.findall(r'(?<![A-Za-z0-9])P0[1-5](?![A-Za-z0-9])', entry['id'])
            if not slots:
                continue
            if len(slots) != 1 or slots[0] in selected:
                raise ValueError('P01-P05 require one unambiguous registry entry per identity')
            selected[slots[0]] = entry
        self.registered_slots = set(selected)
        if not selected:
            raise ValueError('Gen2 requires P01-P05 registry identities')
        super().__init__(calibration, [selected[name] for name in CONFIG['pattern_ids'] if name in selected], extractor)
        self.prepared = {p['id']: TemplateGeometry(p['template'], seed_four=True) for p in self.patterns}
        self.algorithm = dict(module_identity(), config_sha256=digest(CONFIG))
        self.provenance = dict(calibration_sha256=digest(calibration),
            pattern_registry_sha256=digest(patterns), geometry_sha256=digest(self.coordinates.tolist()))
        self.reset()

    def require_online_patterns(self):
        missing = set(CONFIG['pattern_ids'])-self.registered_slots
        if missing:
            raise ValueError('Gen2 requires P01-P05; missing: '+', '.join(sorted(missing)))

    def reset(self):
        self.previous, self.confirmations = {}, {}
        self.roi_motion = ROIMotion()
        self.fast_streak, self.fast_predictions, self.fast_audit = {}, {}, {}
        self.pose_resolver = PoseResolver()
        self.last_stamp = None
        self.discovery_cursor = 0
        self.expand = set()
        self.hints, self.recovery_events = {}, {}
        self.recovery_retry_at = {}
        self.last_was_discovery = False
        self.recording_started = self.next_discovery = None
        self.tracked, self.lost = set(), set()
        self.pending_loss, self.pending_recovery = set(), set()
        self.sweep_reason, self.sweep_remaining, self.sweep_triggers = None, [], set()
        self.sweep_covered = set()
        self.sweep_round = 0
        self.session_seen = set()
        self.session_patterns_frozen = False
        self.startup_seed = {}

    def session_names(self):
        return [p['id'] for p in self.patterns
                if not self.session_patterns_frozen or p['id'] in self.session_seen]

    def session_inventory(self):
        return dict(session_pattern_ids=[p['id'] for p in self.patterns if p['id'] in self.session_seen],
                    session_patterns_frozen=self.session_patterns_frozen)

    def retain_startup_seed(self, result):
        """Expired startup pixels may locate a fresh frame, never confirm a pose."""
        source, completed = result['source_monotonic'], result['completed_monotonic']
        session = self.recording_started
        if (session is None or source is None or result.get('skip_reason') or
                not 0 <= source-session < CONFIG['startup_seconds'] or
                completed-source <= CONFIG['measurement_max_age_seconds']):
            return
        self._lose({p['id'] for p in self.patterns}, clear_hints=True)
        self.startup_seed = {d['pattern_id']: (np.array(d['points_raw']), source, completed)
                             for d in result.get('observations', [])
                             if d['pattern_id'] in self.session_seen}

    def _begin_sweep(self, reason, affected=None):
        names = self.session_names()
        index = self.discovery_cursor % len(names)
        self.sweep_remaining = names[index:]+names[:index]
        if affected is not None:
            self.sweep_remaining = [name for name in self.sweep_remaining if name in affected]
        self.sweep_reason, self.sweep_triggers = reason, {reason}
        self.sweep_covered = set()
        self.sweep_round += 1

    def _request_recovery(self, names, reason):
        pending = self.pending_loss if reason == 'loss' else self.pending_recovery
        for name in names:
            source = self.hints[name][1] if name in self.hints else None
            if name not in self.recovery_events or self.recovery_events[name] != source:
                self.recovery_events[name] = source
                self.recovery_retry_at.pop(name, None)
                pending.add(name)

    def _lose(self, names, clear_hints=False):
        newly_lost = self.tracked.intersection(names)
        self.lost.update(newly_lost)
        self.tracked.difference_update(names)
        for name in names:
            self.previous.pop(name, None)
            self.roi_motion.discard(name, keep_coverage=not clear_hints)
            self.fast_streak.pop(name, None)
            self.confirmations.pop(name, None)
            self.pose_resolver.discard(name)
            if clear_hints:
                self.hints.pop(name, None)
                self.expand.discard(name)
        if clear_hints:
            self._request_recovery(newly_lost, 'loss')

    def pose(self, pattern, raw, recovery=False, refine=True, frozen=True):
        return super().pose(pattern, raw, recovery, refine=refine, frozen=frozen,
                            candidate_evidence=refine, analytic_jacobian=True)

    def _fast_match(self, pattern, candidates, graph, recovery, context, profile):
        name = pattern['id']
        predicted = self.fast_predictions.get(name)
        if predicted is None:
            return None
        audit = self.fast_audit
        if self.fast_streak.get(name, 0) >= CONFIG['roi_association_refresh_frames']:
            audit['forced_checks'].add(name)
            return None
        audit['attempts'] += 1
        context.check()
        order, reason = associate(predicted, candidates)
        if order is not None:
            groups = graph.groups(len(pattern['ids']))
            # Multiple disconnected plausible targets must compete in the original path.
            if len(groups) != 1 or not set(order).issubset(groups[0]):
                order, reason = None, 'competing_groups'
        if order is None:
            audit['fallbacks'] += 1
            audit['reasons'][reason] = audit['reasons'].get(reason, 0)+1
            return None
        raw = np.asarray([candidates[i].center for i in order], dtype=float)
        if not validate_association(self, pattern, raw, recovery, context):
            audit['fallbacks'] += 1
            audit['reasons']['geometry_or_correspondence'] = audit['reasons'].get('geometry_or_correspondence', 0)+1
            return None
        context.check()
        started = time.monotonic()
        value = self.pose(pattern, raw, recovery)
        profile['pose_ms'] += (time.monotonic()-started)*1000
        context.check()
        if value is None:
            audit['fallbacks'] += 1
            audit['reasons']['pose_gate'] = audit['reasons'].get('pose_gate', 0)+1
            return None
        audit['successes'] += 1
        return dict(pattern_id=name, led_ids=pattern['ids'].tolist(), points_raw=raw.tolist(),
                    source='recovery' if recovery else 'baseline',
                    correspondence_search='prediction_guided', **value)

    @timed('matching_total')
    def _matches(self, candidates, recovery, context, profile, active, priority, progress=None,
                 stop_on_first=False):
        if len(candidates) > self.MAX_CANDIDATES:
            return {}, {}, {}
        graph = CandidateGraph(candidates, context,
            [len(p['ids']) for p in self.patterns if p['id'] in active])
        points = np.asarray([c.center for c in candidates], dtype=float).reshape(-1, 2)
        corrected_cache, physical_cache = {}, {}
        outcomes, blocked, searched = {}, {}, {}
        if progress is not None:
            progress.update(outcomes=outcomes, blocked=blocked, searched=searched)
        priority_order = {name: i for i, name in enumerate(priority)} if isinstance(priority, list) else {}
        for pattern in sorted(self.patterns, key=lambda p: (p['id'] not in priority, priority_order.get(p['id'], 0))):
            if pattern['id'] not in active:
                continue
            context.check()
            name, n = pattern['id'], len(pattern['ids'])
            fast = self._fast_match(pattern, candidates, graph, recovery, context, profile)
            if fast is not None:
                outcomes[name], searched[name] = fast, True
                if stop_on_first:
                    break
                continue
            successes, suspicious = {}, False
            if len(points) < n:
                searched[name] = True
                continue
            for group in graph.groups(n):
                context.check()
                key = tuple(group)
                if key not in corrected_cache:
                    corrected_cache[key] = call('matching_undistort', cv2.fisheye.undistortPoints,
                        points[group, None], self.K, self.D, P=self.K)[:, 0]
                corrected = corrected_cache[key]
                if not np.isfinite(corrected).all() or np.abs(corrected).max() > 1e6:
                    continue

                def physical(ordered):
                    cache_key = (name, ordered.tobytes())
                    if cache_key not in physical_cache:
                        context.check()
                        rays = np.column_stack((ordered, np.ones(n))) @ self.inverse.T
                        raw = cv2.fisheye.distortPoints((rays[:, :2]/rays[:, 2:])[:, None],
                                                       self.K, self.D)[:, 0]
                        physical_cache[cache_key] = self.pose(pattern, raw, True, refine=False) is not None
                        context.check()
                    return physical_cache[cache_key]

                status, order = match_pattern(pattern['template'], corrected,
                    tolerance=3. if recovery else 6., inverse_camera=None if recovery else self.inverse,
                    pixel_scale=1. if recovery else 2., geometry_ratio=.025 if recovery else .015,
                    context=context, prepared=self.prepared[name], seed_four=not recovery,
                    validate=physical if recovery else None)
                suspicious |= '歧义' in status
                if order is not None:
                    order = np.asarray(group)[order]
                    successes[tuple(order)] = order
            if len(successes) > 1:
                blocked[name] = 'duplicate_targets'
            elif len(successes) == 1 and not suspicious:
                order = next(iter(successes.values()))
                raw = points[order]
                context.check()
                started = time.monotonic()
                pose = self.pose(pattern, raw, recovery)
                profile['pose_ms'] += (time.monotonic()-started)*1000
                context.check()
                outcomes[name] = (None if pose is None else dict(pattern_id=name,
                    led_ids=pattern['ids'].tolist(), points_raw=raw.tolist(),
                    source='recovery' if recovery else 'baseline', **pose))
            elif suspicious:
                blocked[name] = 'correspondence_ambiguous'
            searched[name] = True
            if stop_on_first and outcomes.get(name) is not None:
                break
        return outcomes, blocked, searched

    def _cpu_signal(self, image, intensity, branch, profile):
        stamp = time.monotonic()
        with span('signal_construct'):
            if branch == 'ordinary':
                signal = cv2.bitwise_and(intensity, cv2.compare(
                    cv2.subtract(intensity, image[:, :, 1]), 25, cv2.CMP_GT))
            elif branch == 'gray':
                signal = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                green, strength = image[:, :, 1].astype(np.float32), intensity.astype(np.float32)
                mask = (strength-green > np.maximum(5, .05*(strength+green+1))).astype(np.uint8)
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool)
                mask &= strength-cv2.blur(strength, (15, 15)) > 3
                signal = np.where(mask, intensity, 0).astype(np.uint8)
                profile['weak_used'] = True
        profile['signal_ms'] += (time.monotonic()-stamp)*1000
        return signal

    def _cpu_branch(self, image, intensity, branch, rectangles, context, profile):
        extractor = RegionExtractor(rectangles, branch == 'weak', context)
        patches = []
        for rectangle in extractor.rectangles:
            context.check()
            x0, y0, x1, y1 = rectangle
            ox, oy, ex, ey = expanded_region(rectangle, image.shape)
            crop = crop_bgr(image, [ox, oy, ex, ey])
            context.check()
            local_intensity = (intensity[oy:ey, ox:ex] if intensity is not None else
                None if branch == 'gray' else
                call('intensity_min_br', cv2.min, crop[:, :, 0], crop[:, :, 2]))
            with span('local_preprocess'):
                signal = self._cpu_signal(crop, local_intensity, branch, profile)
                context.check()
                plane = CPUExtractor()
                stamp = time.monotonic()
                plane.prepare(signal)
                profile['extract_ms'] += (time.monotonic()-stamp)*1000
            patches.append((rectangle, signal[y0-oy:y1-oy, x0-ox:x1-ox],
                            plane.smooth[y0-oy:y1-oy, x0-ox:x1-ox]))
        extractor.prepare_patches(patches, image.shape[:2], max(int(p[2].max()) for p in patches))
        return extractor

    @timed('search_total')
    def search(self, image, rectangles=None, expected=None, deadline=None, priority=None,
               stop_on_first=False, allowed_pattern_ids=None):
        """Unlimited full-frame mode is the offline-core regression entry point."""
        self.validate_image(image)
        started = time.monotonic()
        context = FrameGeometry(deadline)
        rectangles = rectangles or [[0, 0, *self.size]]
        from .methods import merge_rectangles
        rectangles = merge_rectangles(rectangles)
        full = rectangles == [[0, 0, *self.size]]
        registry = {p['id'] for p in self.patterns}
        names = registry if allowed_pattern_ids is None else set(allowed_pattern_ids)
        if not names.issubset(registry):
            raise ValueError('Unknown allowed pattern identity')
        expected = set(expected or names)
        if not expected.issubset(names):
            raise ValueError('Expected identities must be allowed')
        priority = expected if priority is None else priority
        profile = dict(signal_ms=0., extract_ms=0., pose_ms=0., threshold_count=0,
                       candidate_max=0, component_calls=0, moment_calls=0, weak_used=False,
                       match_ms=0., center_ms=0., preprocess_pixels=0, preprocess_regions=0)
        selected, baseline_done, gray_blocked, reasons, searched = {}, set(), set(), {}, set()
        center_cache = ({}, {})
        intensity = None
        exit_reason = 'exhausted' if names else 'no_session_patterns'
        boundary_rectangle = None
        try:
            if names:
                context.check()
            if names and full and not self.gpu:
                intensity = call('intensity_min_br', cv2.min, full_bgr(image)[:, :, 0], full_bgr(image)[:, :, 2])
                context.check()
            if self.gpu and names:
                call('gpu_upload', self.gpu.begin_frame, image, rectangles)
            for branch in ('ordinary', 'gray', 'weak'):
                if exit_reason == 'first_valid' or not names:
                    break
                context.check()
                if expected.issubset(selected):
                    break
                if branch == 'gray' and expected.issubset(baseline_done | gray_blocked):
                    continue
                active = names-set(selected)
                if branch != 'weak':
                    active -= baseline_done
                if branch == 'gray':
                    active -= gray_blocked
                if not active:
                    continue
                closed, last_ambiguous = set(), set()
                stamp = time.monotonic()
                if self.gpu and branch != 'weak':
                    pixels = sum((r[2]-r[0]+4)*(r[3]-r[1]+4) for r in rectangles)
                    local = pixels < image.shape[0]*image.shape[1]
                    profile['preprocess_pixels'] += pixels if local else image.shape[0]*image.shape[1]
                    profile['preprocess_regions'] += len(rectangles) if local else 1
                else:
                    for rectangle in rectangles:
                        x0, y0, x1, y1 = expanded_region(rectangle, image.shape)
                        profile['preprocess_pixels'] += (x1-x0)*(y1-y0)
                        profile['preprocess_regions'] += 1
                if self.gpu and branch != 'weak':
                    extractor = GPURegionExtractor(self.gpu, branch, rectangles, context,
                        self.gpu_components)
                    profile['extract_ms'] += (time.monotonic()-stamp)*1000
                else:
                    if intensity is None and full:
                        intensity = call('intensity_min_br', cv2.min, full_bgr(image)[:, :, 0], full_bgr(image)[:, :, 2])
                    extractor = self._cpu_branch(image, intensity, branch, rectangles, context, profile)
                thresholds = CONFIG['weak_thresholds' if branch == 'weak' else
                                    'gray_thresholds' if branch == 'gray' else 'ordinary_thresholds']
                for threshold in thresholds:
                    context.check()
                    stamp = time.monotonic()
                    before = extractor.calls, extractor.moment_calls
                    candidates = extractor.extract(threshold, 1 if branch == 'weak' else 2)
                    profile['extract_ms'] += (time.monotonic()-stamp)*1000
                    profile['component_calls'] += extractor.calls-before[0]
                    profile['moment_calls'] += extractor.moment_calls-before[1]
                    profile['threshold_count'] += 1
                    profile['candidate_max'] = max(profile['candidate_max'], len(candidates))
                    context.check()
                    pending = active-closed-set(selected)
                    if not pending:
                        break
                    # Only the FINAL color status suppresses gray; a later threshold
                    # may remove a suspicious group. Duplicate targets close a branch.
                    last_ambiguous -= pending
                    for name in pending:
                        reasons.pop(name, None)
                    if not candidates or len(candidates) > self.MAX_CANDIDATES:
                        continue
                    if branch == 'weak':
                        stamp = time.monotonic()
                        candidates = recovery_centers(candidates, intensity, context, center_cache, image)
                        profile['center_ms'] += (time.monotonic()-stamp)*1000
                        context.check()
                    stamp = time.monotonic()
                    progress, interrupted = {}, False
                    try:
                        outcomes, blocked, visited = self._matches(
                            candidates, branch == 'weak', context, profile, pending, priority, progress,
                            **(dict(stop_on_first=True) if stop_on_first else {}))
                    except SearchBudgetExceeded:
                        outcomes, blocked, visited = (progress.get(key, {})
                            for key in ('outcomes', 'blocked', 'searched'))
                        interrupted = True
                    finally:
                        profile['match_ms'] += (time.monotonic()-stamp)*1000
                    searched.update(visited)
                    last_ambiguous.update(blocked)
                    closed.update(name for name, reason in blocked.items() if reason == 'duplicate_targets')
                    reasons.update(blocked)
                    for name, value in outcomes.items():
                        if name in selected or (branch != 'weak' and name in baseline_done):
                            continue
                        if branch == 'gray' and name in gray_blocked:
                            continue
                        closed.add(name)
                        if branch != 'weak':
                            baseline_done.add(name)
                        if value is not None:
                            reasons.pop(name, None)
                            value['threshold'] = threshold
                            selected[name] = value
                    if interrupted:
                        raise SearchBudgetExceeded('incomplete pattern search')
                    if stop_on_first and reject_conflicts(list(selected.values()))[0]:
                        exit_reason = 'first_valid'
                        break
                    if expected.issubset(selected):
                        exit_reason = 'expected_ids'
                        break
                if branch == 'ordinary':
                    gray_blocked = last_ambiguous
        except SearchBudgetExceeded:
            exit_reason = 'budget'
        except RegionBoundary as exc:
            exit_reason = 'roi_boundary'
            boundary_rectangle = exc.rectangle
        except Exception:
            if self.gpu:
                self.reset()
            raise
        accepted, conflicts = call('identity_conflicts', reject_conflicts, list(selected.values()))
        conflicts = sorted(set(conflicts) | {name for name, reason in reasons.items()
            if 'ambiguous' in reason or reason == 'duplicate_targets'})
        completed = names if exit_reason == 'exhausted' else set(selected)
        profile.update(context.stats)
        elapsed = (time.monotonic()-started)*1000
        return dict(schema='dohc.infrared.v1', method='gen2', detections=accepted,
            ambiguous_pattern_ids=conflicts, algorithm=self.algorithm, provenance=self.provenance,
            backend=self.backend, image_size=list(self.size), processing_ms=elapsed, stages=profile,
            search=dict(rois=rectangles, searched_pattern_ids=sorted(searched),
                not_searched=sorted(names-searched), allowed_pattern_ids=sorted(names), reasons=reasons,
                completed_pattern_ids=sorted(completed), incomplete_pattern_ids=sorted(names-completed),
                exit_reason=exit_reason, boundary_rectangle=boundary_rectangle,
                budget_exhausted=exit_reason == 'budget',
                budget_overrun_ms=0. if deadline is None else max(0., (time.monotonic()-deadline)*1000)))

    def _retry_static_roi(self, image, result, roi_by_name, prediction, stamp, deadline, options):
        """Prediction must not force recovery when the original ROI still works."""
        predicted = {name for name, audit in prediction.items() if audit['used']}
        if not predicted:
            return result
        observed = {d['pattern_id'] for d in result['detections']}
        completed = set(result['search'].get('completed_pattern_ids',
            roi_by_name if result['search']['exit_reason'] == 'exhausted' else observed))
        failed = predicted & (completed-observed)
        if result['search']['exit_reason'] == 'roi_boundary':
            failed |= predicted-observed
        if (not failed or result['search']['budget_exhausted'] or
                time.monotonic() >= deadline or result['ambiguous_pattern_ids']):
            return result
        static = []
        for name, rectangle in roi_by_name.items():
            if name not in predicted:
                static.append(rectangle)
                continue
            points, old = self.hints[name]
            margin = max(12., .5*np.linalg.norm(np.ptp(points, axis=0)))+120*(stamp-old)
            if name in self.expand:
                margin *= 2
            lo = np.maximum(np.floor(points.min(0)-margin), 0).astype(int)
            hi = np.minimum(np.ceil(points.max(0)+margin+1), self.size).astype(int)
            static.append([int(v) for v in (*lo, *hi)])
        from .methods import merge_rectangles
        if merge_rectangles(static) == result['search']['rois']:
            return result
        predicted_search = result['search']
        # Retry original matching, never a prediction-guided shortcut.
        fast = getattr(self, 'fast_predictions', None)
        if fast is not None:
            self.fast_predictions = {}
        try:
            retry = self.search(image, static, set(roi_by_name), deadline=deadline, **options)
        finally:
            if fast is not None:
                self.fast_predictions = fast
        retry_names = set(retry['search']['completed_pattern_ids'])
        found = {d['pattern_id']: d for d in result['detections'] if d['pattern_id'] not in retry_names}
        found.update({d['pattern_id']: d for d in retry['detections']})
        accepted, conflicts = reject_conflicts(list(found.values()))
        retry['detections'] = accepted
        retry['ambiguous_pattern_ids'] = sorted(set(retry['ambiguous_pattern_ids']) | set(conflicts))
        retry['search']['completed_pattern_ids'] = sorted(completed | retry_names)
        retry['search']['incomplete_pattern_ids'] = sorted(set(roi_by_name)-(completed | retry_names))
        retry['search']['static_roi_retry'] = dict(reason='prediction_miss_or_boundary',
            predicted_search=predicted_search, retry_rois=static,
            recovered_pattern_ids=sorted({d['pattern_id'] for d in retry['detections']}-observed))
        for key, value in result['stages'].items():
            if isinstance(value, bool):
                retry['stages'][key] = value or retry['stages'].get(key, False)
            elif isinstance(value, (int, float)):
                retry['stages'][key] = retry['stages'].get(key, 0)+value
        return retry

    @profiled
    def detect(self, image, timestamp=None, recording_started_monotonic=None):
        started = time.monotonic()
        stamp = started if timestamp is None else float(timestamp)
        if not np.isfinite(stamp):
            raise ValueError('finite monotonic frame time required')
        session = recording_started_monotonic
        if session is not None and (isinstance(session, bool) or not np.isfinite(session) or not 0 <= session <= stamp):
            raise ValueError('recording start must precede the source frame on the same monotonic clock')
        if session is not None:
            self.require_online_patterns()
        if session != self.recording_started or self.last_stamp is not None and stamp < self.last_stamp:
            self.reset()
            self.recording_started = session
            if session is not None:
                self.next_discovery = session+CONFIG['startup_seconds']+CONFIG['discovery_interval_seconds']
        startup_deadline = None if session is None else session+CONFIG['startup_seconds']
        startup = session is not None and max(stamp, started) < startup_deadline
        if session is not None and not startup:
            self.session_patterns_frozen = True
        if self.last_stamp == stamp:
            return dict(schema='dohc.infrared.v1', method='gen2', detections=[], observations=[],
                        ambiguous_pattern_ids=[], algorithm=self.algorithm, provenance=self.provenance,
                        backend=self.backend, image_size=list(self.size), processing_ms=0.,
                        quality=dict(state='NO_NEW_FRAME'), search_mode='duplicate_source_frame',
                        **self.session_inventory())
        names = self.session_names()
        with span('schedule_and_search'):
            self.last_stamp = stamp
            self._lose({name for name, old in self.previous.items()
                        if not 0 < stamp-old[1] <= CONFIG['measurement_max_age_seconds']})
            self.confirmations = {name: old for name, old in self.confirmations.items()
                                  if 0 < stamp-old[0] <= CONFIG['confirmation_seconds']}
            roi_by_name = {}
            roi_prediction, self.predicted_points = {}, {}
            for name, (points, old_stamp) in self.hints.items():
                if name not in names:
                    continue
                if not 0 < stamp-old_stamp <= CONFIG['roi_hint_max_age_seconds']:
                    self._request_recovery({name}, 'loss')
                    self.expand.discard(name)
                    continue
                rectangle, predicted, audit = self.roi_motion.rectangle(
                    name, points, old_stamp, stamp, self.size, name in self.expand)
                roi_by_name[name] = rectangle
                roi_prediction[name] = audit
                if predicted is not None:
                    self.predicted_points[name] = predicted
            seed, self.startup_seed = self.startup_seed, {}
            bootstrap = False
            for name, (points, source, completed) in seed.items():
                if (name not in names or stamp <= source or
                        not 0 <= started-completed <= CONFIG['measurement_max_age_seconds']):
                    continue
                margin = max(12., .5*np.linalg.norm(np.ptp(points, axis=0)))
                margin += min(480., 120*(stamp-source))
                lo = np.maximum(np.floor(points.min(0)-margin), 0).astype(int)
                hi = np.minimum(np.ceil(points.max(0)+margin+1), self.size).astype(int)
                roi_by_name[name] = [int(v) for v in (*lo, *hi)]
                bootstrap = True
            rectangles = list(roi_by_name.values())
            expanded = self.expand.intersection(roi_by_name)
            reason, priority_identity = None, None
            retry_due = set()
            if names and startup:
                if self.sweep_reason != 'startup' or not self.sweep_remaining:
                    self._begin_sweep('startup')
            elif names:
                if self.sweep_reason == 'startup':
                    self.sweep_reason, self.sweep_remaining, self.sweep_triggers = None, [], set()
                # Event deduplication must not suppress later recovery opportunities.
                retry_due = {name for name, due in self.recovery_retry_at.items()
                    if name in names and name in self.recovery_events and stamp >= due
                    and name not in self.sweep_remaining}
                self.pending_recovery.update(retry_due)
                due = session is not None and stamp >= self.next_discovery
                if due:
                    interval = CONFIG['discovery_interval_seconds']
                    self.next_discovery += (int((stamp-self.next_discovery)//interval)+1)*interval
                for trigger, affected in (('loss', self.pending_loss),
                                          ('roi_recovery', self.pending_recovery)):
                    affected = affected.intersection(names)
                    if not affected:
                        continue
                    if self.sweep_reason is None:
                        self._begin_sweep(trigger, affected)
                    else:
                        self.sweep_triggers.add(trigger)
                        self.sweep_covered.difference_update(affected)
                        self.sweep_remaining.extend(name for name in names
                            if name in affected and name not in self.sweep_remaining)
                if due:
                    if self.sweep_reason is None:
                        self._begin_sweep('periodic')
                    else:
                        self.sweep_triggers.add('periodic')
                        self.sweep_remaining.extend(name for name in names
                            if name not in self.sweep_covered and name not in self.sweep_remaining)
                if self.sweep_reason is None and not rectangles:
                    self._begin_sweep('searching')
            roi_coverage = {name: dict(enabled=CONFIG['roi_coverage_enabled'], applied=False,
                reason='startup_or_discovery', original_roi=r, roi=r, area_ratio=1.)
                for name, r in roi_by_name.items()}
            self.fast_predictions = {}
            self.fast_audit = dict(attempts=0, successes=0, fallbacks=0, reasons={}, forced_checks=set())
            self.pending_loss.clear()
            self.pending_recovery.clear()
            discovery = bool(names) and not bootstrap and (startup or (self.sweep_reason is not None and
                (not rectangles or not self.last_was_discovery) and not expanded))
            if not names:
                self.sweep_reason, self.sweep_remaining, self.sweep_triggers = None, [], set()
                result = self.search(image, allowed_pattern_ids=[])
                mode, reason = 'inactive', 'no_session_patterns'
            elif discovery:
                self.fast_predictions = {}
                reason = self.sweep_reason
                priority_identity = self.sweep_remaining.pop(0)
                index = names.index(priority_identity)
                priority = names[index:]+names[:index]
                self.discovery_cursor = (index+1) % len(names)
                result = self.search(image, expected=set(names), priority=priority,
                                     deadline=startup_deadline if startup else started+CONFIG['discovery_budget_seconds'],
                                     stop_on_first=not startup, allowed_pattern_ids=names)
                mode = 'discovery'
            else:
                if not startup and not bootstrap:
                    roi_by_name, roi_coverage = self.roi_motion.cover_rectangles(
                        roi_by_name, roi_prediction, stamp, self.size)
                    rectangles = list(roi_by_name.values())
                self.fast_predictions = ({name: points for name, points in self.predicted_points.items()
                    if name not in expanded} if CONFIG['roi_association_enabled'] and not startup and not bootstrap else {})
                options = {}
                if session is not None:
                    index = self.discovery_cursor % len(names)
                    options['priority'] = names[index:]+names[:index]
                    self.discovery_cursor += 1
                deadline = (min(startup_deadline, started+CONFIG['tracking_budget_seconds'])
                            if startup else started+CONFIG['tracking_budget_seconds'])
                result = self.search(image, rectangles, set(roi_by_name), deadline=deadline,
                                     allowed_pattern_ids=names, **options)
                if not bootstrap:
                    result = self._retry_static_roi(image, result, roi_by_name, roi_prediction, stamp,
                        deadline, dict(allowed_pattern_ids=names, **options))
                mode = 'roi'
                if bootstrap:
                    reason = 'startup_roi'
            self.last_was_discovery = mode == 'discovery'
        self.fast_predictions = {}
        result['search']['fast_association'] = dict(self.fast_audit,
            forced_checks=sorted(self.fast_audit['forced_checks']))
        for audit in roi_prediction.values():
            audit['applied'] = bool(mode == 'roi' and not bootstrap and audit['used'])
        result['search']['roi_prediction'] = roi_prediction
        result['search']['roi_coverage'] = roi_coverage
        for name in result['ambiguous_pattern_ids']:
            self._lose({name}, clear_hints=True)
        observations = result['detections']
        observed = {d['pattern_id'] for d in observations}
        if startup:
            self.session_seen.update(observed)
        completed = set(result['search'].get('completed_pattern_ids',
            names if result['search']['exit_reason'] == 'exhausted' else observed))
        if mode == 'discovery':
            self.sweep_covered.update(completed | {priority_identity})
            self.sweep_remaining = [name for name in self.sweep_remaining if name not in self.sweep_covered]
            recovery_attempted = (completed | {priority_identity}).intersection(self.recovery_events)
            for name in recovery_attempted:
                self.recovery_retry_at[name] = stamp+CONFIG['roi_recovery_retry_seconds']
        else:
            recovery_attempted = set()
        self._lose((completed if mode == 'discovery' else completed.intersection(roi_by_name))-observed)
        self.pose_resolver.retain(observed | (set(self.previous)-
            set(result['search'].get('completed_pattern_ids', observed))), stamp)
        confirmed = []
        for detection in observations:
            with span('identity_confirmation'):
                name = detection['pattern_id']
                points = np.array(detection['points_raw'])
                old = self.confirmations.get(name)
                count = 1
                if old is not None and np.linalg.norm(points.mean(0)-old[2]) <= max(
                        12., np.linalg.norm(np.ptp(points, axis=0))):
                    count = old[1]+1
                self.confirmations[name] = (stamp, count, points.mean(0))
                identity_ok = count >= CONFIG['confirmation_frames']
            self.pose_resolver.resolve_dual(detection, stamp, identity_ok,
                search_complete=detection.get('correspondence_search') != 'prediction_guided' and
                    not result['search']['budget_exhausted'] and
                    result['search']['exit_reason'] not in ('roi_boundary', 'first_valid'))
            detection['quality'] = dict(identity_confirmed=identity_ok, measurement_fresh=True,
                pose_available=identity_ok and detection.get('pose') is not None,
                pose_usable=detection['pose_resolution']['output_supported'],
                orientation_supported=detection['pose_resolution']['orientation_supported'],
                position_usable=identity_ok and 'position' in detection,
                state='TRACKING' if identity_ok else 'TENTATIVE')
            if identity_ok:
                confirmed.append(detection)
                self.tracked.add(name)
                self.lost.discard(name)
            self.fast_streak[name] = (self.fast_streak.get(name, 0)+1
                if detection.get('correspondence_search') == 'prediction_guided' else 0)
            self.previous[name] = (detection, stamp)
            self.hints[name] = (points.copy(), stamp)
            self.roi_motion.commit(name, points, detection.get('led_ids', []), stamp,
                                   time.monotonic(), identity_ok and detection.get('pose') is not None)
            self.expand.discard(name)
            self.recovery_events.pop(name, None)
            self.recovery_retry_at.pop(name, None)
            self.pending_loss.discard(name)
            self.pending_recovery.discard(name)
        if mode == 'roi':
            affected = completed.intersection(roi_by_name)-observed
            if result['search']['exit_reason'] == 'roi_boundary':
                boundary = result['search'].get('boundary_rectangle')
                affected.update(name for name, r in roi_by_name.items() if name not in observed and
                    (boundary is None or r[0] < boundary[2] and boundary[0] < r[2] and
                     r[1] < boundary[3] and boundary[1] < r[3]))
            failed_expansion = affected & expanded
            self._request_recovery(failed_expansion, 'roi_recovery')
            self.expand.difference_update(failed_expansion)
            self.expand.update(name for name in affected-expanded
                if name not in self.recovery_events)
        result['search']['recovery_retry'] = dict(interval_seconds=CONFIG['roi_recovery_retry_seconds'],
            scheduled_pattern_ids=sorted(retry_due), attempted_pattern_ids=sorted(recovery_attempted),
            next_source_monotonic=dict(self.recovery_retry_at))
        if session is not None:
            result.update(search_reason=reason or ('expanded_roi' if expanded else 'tracking'),
                          search_round=self.sweep_round,
                          search_triggers=sorted(self.sweep_triggers) if reason else [],
                          priority_pattern_id=priority_identity,
                          round_remaining_pattern_ids=list(self.sweep_remaining),
                          round_complete=bool(reason) and not self.sweep_remaining,
                          tracked_pattern_ids=sorted(self.tracked), lost_pattern_ids=sorted(self.lost),
                          next_discovery_monotonic=self.next_discovery)
        if reason and not self.sweep_remaining:
            self.sweep_reason, self.sweep_triggers = None, set()
        state = ('NO_SESSION_TARGETS' if not names else
                 'TRACKING' if confirmed else 'TENTATIVE' if observations else
                 'RECOVERING' if self.previous else 'SEARCHING')
        if session is not None and time.monotonic() >= startup_deadline:
            self.session_patterns_frozen = True
        result.update(detections=confirmed, observations=observations, search_mode=mode,
                      quality=dict(state=state), processing_ms=(time.monotonic()-started)*1000,
                      **self.session_inventory())
        return result
