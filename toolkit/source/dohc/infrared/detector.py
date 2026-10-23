"""4K LED extraction, multi-pattern identity and fisheye IPPE + G refinement.

No motion prediction: every published detection belongs to this input image.
The caller supplies an explicit camera calibration and pattern registry.
"""
import csv
import json
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from .profiling import active, call, timed, note
from scipy.optimize import least_squares

from .geometry import candidate_groups, match_pattern

Candidate = namedtuple("Candidate", "center peak core_area")


def _g_problem(objects, raw, camera, distortion, analytic, measured):
    counts = dict(solver_residual_calls=0, solver_jacobian_calls=0,
                  solver_projection_calls=0, solver_cache_hits=0, solver_projection_ms=0.) if measured else None
    object_points = objects[:, None]
    cached_x = cached_residual = cached_jacobian = None

    def evaluate(x):
        nonlocal cached_x, cached_residual, cached_jacobian
        if analytic and cached_x is not None and np.array_equal(x, cached_x):
            if measured:
                counts['solver_cache_hits'] += 1
            return cached_residual, cached_jacobian
        started = time.perf_counter_ns() if measured else None
        projected, full_jacobian = cv2.fisheye.projectPoints(object_points, x[:3], x[3:], camera, distortion)
        if measured:
            counts['solver_projection_calls'] += 1
            counts['solver_projection_ms'] += (time.perf_counter_ns()-started)/1e6
        residual = (projected[:, 0]-raw).ravel()
        if not analytic:
            return residual, None
        if full_jacobian.shape != (raw.size, 15):
            raise ValueError('Expected OpenCV fisheye 2N x 15 Jacobian')
        cached_x = x.copy()
        cached_residual = residual
        cached_jacobian = np.ascontiguousarray(full_jacobian[:, 8:14])
        return cached_residual, cached_jacobian

    def residual(x):
        if measured:
            counts['solver_residual_calls'] += 1
        value = evaluate(x)[0]
        # SciPy soft_l1 scales both residual and Jacobian in place.
        return value.copy() if analytic else value

    def jacobian(x):
        if measured:
            counts['solver_jacobian_calls'] += 1
        return evaluate(x)[1].copy()

    return residual, jacobian, counts


def _refine_g(objects, raw, camera, distortion, initial, index, analytic=False):
    measured = active()
    residual, jacobian, counts = _g_problem(objects, raw, camera, distortion, analytic, measured)
    started = time.perf_counter_ns() if measured else None
    opt = call('g_candidate_'+str(index), least_squares, residual, initial,
               jac=jacobian if analytic else '2-point', loss='soft_l1', f_scale=.3,
               x_scale=[1, 1, 1, .1, .1, 1], max_nfev=60)
    if measured:
        elapsed = (time.perf_counter_ns()-started)/1e6
        errors = np.linalg.norm(opt.fun.reshape(-1, 2), axis=1)
        note('g_optimization', dict(candidate=index, jacobian_mode='analytic' if analytic else '2-point',
            nfev=opt.nfev, njev=opt.njev, iterations=None, iteration_count_available=False,
            success=bool(opt.success), status=int(opt.status), cost=float(opt.cost),
            final_rms_px=float(np.sqrt(np.mean(errors**2))), final_max_error_px=float(errors.max()),
            robust_cost=float(np.sum(np.sqrt(1+(opt.fun/.3)**2)-1)),
            g_wall_ms=elapsed, **counts))
    return opt


def coordinates():
    path = Path(__file__).with_name("led_coordinates.csv")
    points = np.full((100, 2), np.nan)
    with path.open(encoding="gb18030", newline="") as stream:
        for _, row, col, x, y in list(csv.reader(stream))[1:]:
            row, col = int(row), int(col)
            if not (1 <= row <= 10 and 1 <= col <= 10):
                raise ValueError("LED row/column out of range")
            index = (row - 1) * 10 + col - 1
            if np.isfinite(points[index]).any():
                raise ValueError("Duplicate LED coordinate")
            points[index] = float(x), float(y)
    if not np.isfinite(points).all():
        raise ValueError("Incomplete LED coordinates")
    return points


class CPUExtractor:
    name = "cpu"
    is_gpu = False
    is_npu = False

    def stats(self):
        return {"gpu_calls": 0, "npu_calls": 0, "cpu_fallback": 0}

    @timed('extract_prepare')
    def prepare(self, gray, weights=None):
        self.gray = gray
        self.weights = gray if weights is None else weights
        self.smooth = call('gaussian_smoothing', cv2.GaussianBlur, gray, (0, 0), .6)

    def extract(self, threshold, min_area=2):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (self.smooth > threshold).astype(np.uint8), connectivity=8)
        result = []
        for label in range(1, count):
            x, y, w, h, area = map(int, stats[label])
            if area < min_area:
                continue
            patch = self.weights[y:y+h, x:x+w]
            mask = labels[y:y+h, x:x+w] == label
            weights = np.where(mask, np.maximum(patch.astype(np.int32) - threshold + 1, 0), 0)
            total = weights.sum(dtype=np.int64)
            if not total:
                continue
            yy, xx = np.indices(patch.shape)
            center = np.float32([x + (xx * weights).sum() / total,
                                 y + (yy * weights).sum() / total])
            result.append(Candidate(center, int(patch[mask].max()), area))
            if len(result) > 2048:
                return result  # Caller rejects the entire over-budget set.
        result.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
        return result


def refine_centers(candidates, intensity):
    """Frozen weak-recovery annulus correction, bounded to 1.5 original pixels."""
    if not candidates:
        return candidates
    points = np.array([c.center for c in candidates])
    distance = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    radii = np.clip(np.floor(distance.min(1) * .35), 2, 6).astype(int)
    rebuilt = []
    for candidate, radius in zip(candidates, radii):
        x, y = map(float, candidate.center)
        outer = radius + 2
        x0, y0 = max(0, int(np.floor(x-outer))), max(0, int(np.floor(y-outer)))
        x1 = min(intensity.shape[1], int(np.ceil(x+outer))+1)
        y1 = min(intensity.shape[0], int(np.ceil(y+outer))+1)
        patch = intensity[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.indices(patch.shape, dtype=np.float32)
        xx += x0
        yy += y0
        d2 = (xx-x)**2 + (yy-y)**2
        ring = (d2 > radius**2) & (d2 <= outer**2)
        if ring.any():
            weights = np.maximum(patch - np.median(patch[ring]), 0) * (d2 <= radius**2)
            total = float(weights.sum())
            if total > 0:
                center = np.float32([(weights*xx).sum()/total, (weights*yy).sum()/total])
                if np.linalg.norm(center-candidate.center) <= 1.5:
                    candidate = Candidate(center, candidate.peak, candidate.core_area)
        rebuilt.append(candidate)
    return rebuilt


def overlapping(first, second):
    """A light cannot establish two tracker identities in the same image."""
    a, b = np.array(first["points_raw"]), np.array(second["points_raw"])
    return bool(np.any(np.linalg.norm(a[:, None] - b[None], axis=2) < 2.))


def reject_conflicts(detections):
    rejected = set()
    for i, first in enumerate(detections):
        for j in range(i):
            second = detections[j]
            if first["pattern_id"] == second["pattern_id"] or overlapping(first, second):
                rejected.update((i, j))
    return ([d for i, d in enumerate(detections) if i not in rejected],
            sorted({detections[i]["pattern_id"] for i in rejected}))


class MultiTracker:
    MAX_CANDIDATES = 256
    MAX_PROCESSING_SECONDS = 2.0

    def __init__(self, calibration, patterns, extractor=None):
        self.K = np.asarray(calibration["K"], dtype=np.float64)
        self.D = np.asarray(calibration["D"], dtype=np.float64).reshape(-1)
        self.size = (calibration["image_width"], calibration["image_height"])
        if (calibration.get("camera_model") != "opencv_fisheye"
                or self.K.shape != (3, 3) or self.D.shape != (4,)
                or not np.isfinite(self.K).all() or not np.isfinite(self.D).all()
                or min(self.K[0, 0], self.K[1, 1]) <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or self.size != (3840, 2160)):
            raise ValueError("Explicit native 3840x2160 fisheye calibration required")
        self.inverse = np.linalg.inv(self.K)
        self.coordinates = coordinates()
        self.origin = self.coordinates.mean(0)
        self.patterns = []
        if not 1 <= len(patterns) <= 5:
            raise ValueError("Supply one to five explicit tracker patterns")
        for entry in patterns:
            frame = entry["frame"]
            name = entry["id"]
            if (not isinstance(name, str) or not name.strip()
                    or any(p["id"] == name for p in self.patterns)
                    or len(frame) != 100
                    or any(type(v) is not int or not 0 <= v <= 255 for v in frame)):
                raise ValueError("Unique pattern ID and 100 integer LED values required")
            ids = np.flatnonzero(np.array(frame) > 0)
            if not 5 <= len(ids) <= 12:
                raise ValueError("This G port requires 5–12 separate LEDs, not luminous blocks")
            if any(np.array_equal(ids, p["ids"]) for p in self.patterns):
                raise ValueError("Two identities cannot use the same illuminated LEDs")
            template = self.coordinates[ids]
            if len(cv2.convexHull(template.astype(np.float32))) < 4:
                raise ValueError("Pattern requires at least four hull vertices")
            self.patterns.append(dict(id=name, ids=ids, template=template,
                objects=np.column_stack(((template-self.origin)/1000, np.zeros(len(ids))))))
        self.extractor = extractor or CPUExtractor()

    @timed('pose_total')
    def pose(self, pattern, raw, recovery=False, refine=True, frozen=False,
             candidate_evidence=False, analytic_jacobian=False):
        objects = pattern["objects"]
        normalized = call('pose_undistort', cv2.fisheye.undistortPoints, raw[:, None], self.K, self.D)
        try:
            solved = call('u_ippe', cv2.solvePnPGeneric, objects, normalized, np.eye(3), np.zeros(4),
                                         flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        limits = (.5, 1.) if recovery else (2., 3.)

        def residual(x):
            return (cv2.fisheye.projectPoints(objects[:, None], x[:3], x[3:],
                                             self.K, self.D)[0][:, 0] - raw).ravel()

        @timed('pose_validation')
        def accepted(x, gates=limits, residual_value=None):
            if not np.isfinite(x).all():
                return None
            if np.min((objects @ cv2.Rodrigues(x[:3])[0].T + x[3:])[:, 2]) <= 0:
                return None
            err = np.linalg.norm((residual(x) if residual_value is None else residual_value).reshape(-1, 2), axis=1)
            rms = float(np.sqrt(np.mean(err**2)))
            if not np.isfinite(err).all() or rms > gates[0] or err.max() > gates[1]:
                return None
            return dict(rvec=x[:3].tolist(), tvec_m=x[3:].tolist(), rms_px=rms,
                        max_error_px=float(err.max()))

        candidates = []
        for rvec, tvec in zip(solved[1], solved[2]):
            value = accepted(np.r_[rvec.ravel(), tvec.ravel()], (2., 3.) if frozen else limits)
            if value is not None:
                if frozen and any(np.linalg.norm(tvec.ravel()-p['tvec_m']) < 1e-5 and
                        np.degrees(np.arccos(np.clip((np.trace(cv2.Rodrigues(rvec)[0] @
                            cv2.Rodrigues(np.array(p['rvec']))[0].T)-1)/2, -1, 1))) < .1
                        for p in candidates):
                    continue
                candidates.append(value)
        candidates.sort(key=lambda p: p["rms_px"])
        if not candidates:
            return None
        # Freeze correspondence before G; no temporal or mocap selection.
        best = candidates[0]
        if best['rms_px'] > limits[0] or best['max_error_px'] > limits[1]:
            return None
        def cost(x, residual_value=None):
            value = residual(x) if residual_value is None else residual_value
            return float(np.sum(np.sqrt(1+(value/.3)**2)-1))
        best_cost = cost(np.r_[best["rvec"], best["tvec_m"]])
        refined, measurements = [], []
        for index, value in enumerate(candidates if refine else ()):
            initial = np.r_[value["rvec"], value["tvec_m"]]
            opt = _refine_g(objects, raw, self.K, self.D, initial, index, analytic_jacobian)
            final_residual = opt.fun if analytic_jacobian else None
            pose = accepted(opt.x, residual_value=final_residual) if opt.success else None
            final_cost = cost(opt.x, final_residual) if pose is not None else None
            if pose is not None:
                refined.append((final_cost, pose))
            if candidate_evidence:
                # Keep each current-image branch for Gen2's causal selector.
                # A failed/worse G fit may only fall back to a U fit that passes
                # this observation's gates (including weak-recovery gates).
                own_cost = cost(initial)
                use_g = pose is not None and final_cost <= own_cost + 1e-9
                measurement = pose if use_g else accepted(initial)
                if measurement is not None:
                    measurements.append(dict(pose=measurement, u_candidate_index=index,
                        refinement_applied=use_g, candidate_stage='G' if use_g else 'U',
                        robust_cost=final_cost if use_g else own_cost))
        applied = False
        if refined:
            refined.sort(key=lambda p: p[0])
            if refined[0][0] <= best_cost + 1e-9:
                best_cost, best = refined[0]
                applied = True
        result = dict(pose=best, pose_candidates=candidates,
                    pose_ambiguous=len(candidates) > 1 and
                    candidates[1]["rms_px"]-candidates[0]["rms_px"] < .25,
                    refinement_applied=applied, robust_cost=best_cost,
                    candidate_stage="U")
        if candidate_evidence and refine:
            result['pose_measurements'] = measurements
        return result

    def match_candidates(self, candidates, recovery=False, skip_pattern_ids=None):
        if len(candidates) > self.MAX_CANDIDATES:
            return []
        skip_pattern_ids = set(skip_pattern_ids or ())
        points = np.asarray([c.center for c in candidates], dtype=float).reshape(-1, 2)
        found = []
        for pattern in self.patterns:
            if pattern["id"] in skip_pattern_ids:
                continue
            n = len(pattern["ids"])
            if len(points) < n:
                continue
            matches = {}
            for group in candidate_groups(candidates, n):
                corrected = cv2.fisheye.undistortPoints(
                    points[group, None], self.K, self.D, P=self.K)[:, 0]
                if not np.isfinite(corrected).all() or np.abs(corrected).max() > 1e6:
                    continue
                _, orders = match_pattern(pattern["template"], corrected,
                    tolerance=3. if recovery else 6.,
                    inverse_camera=None if recovery else self.inverse,
                    pixel_scale=1. if recovery else 2., all_orders=True,
                    geometry_ratio=.025 if recovery else .015)
                for order in orders:
                    key = tuple(np.asarray(group)[order])
                    if key in matches:
                        continue
                    raw = points[list(key)]
                    pose = self.pose(pattern, raw, recovery)
                    if pose is not None:
                        matches[key] = dict(pattern_id=pattern["id"], led_ids=pattern["ids"].tolist(),
                            points_raw=raw.tolist(), source="recovery" if recovery else "baseline", **pose)
            # Refuse ambiguous point assignments, including multiple copies of one ID.
            if len(matches) == 1:
                found.extend(matches.values())
            elif matches:
                # Retain all plausible correspondences for cross-pattern rejection.
                found.extend(matches.values())
        return found

    def validate_image(self, image):
        if image.dtype != np.uint8 or image.shape != (self.size[1], self.size[0], 3):
            raise ValueError("Expected a native 3840x2160 uint8 BGR frame")

    def _detect_branches(self, image, branches, intensity, method="baseline", timestamp=None, min_area=None):
        self.validate_image(image)
        start = time.monotonic()
        before = self.extractor.stats() if hasattr(self.extractor, "stats") else {}
        deadline = start + self.MAX_PROCESSING_SECONDS
        selected, ambiguous = {}, set()
        stages = dict(prepare_ms=0., extract_ms=0., match_ms=0., threshold_count=0,
                      first_match_ms=None, exit_reason="exhausted")
        expected = getattr(self, "tracking_pattern_ids", None)
        def finished():
            if expected is not None:
                return expected.issubset(selected) and not ambiguous
            return len(selected) == len(self.patterns)
        for branch in branches:
            signal, thresholds, recovery = branch[:3]
            weights = branch[3] if len(branch) == 4 else None
            if time.monotonic() >= deadline:
                stages["exit_reason"] = "budget"
                break
            stage_start = time.monotonic()
            self.extractor.prepare(signal, weights) if weights is not None else self.extractor.prepare(signal)
            stages["prepare_ms"] += (time.monotonic()-stage_start)*1000
            for threshold in thresholds:
                if time.monotonic() >= deadline:
                    stages["exit_reason"] = "budget"
                    break
                stage_start = time.monotonic()
                candidates = self.extractor.extract(threshold, min_area if min_area is not None else (1 if recovery else 2))
                stages["extract_ms"] += (time.monotonic()-stage_start)*1000
                stages["threshold_count"] += 1
                if not candidates or len(candidates) > self.MAX_CANDIDATES:
                    continue
                stage_start = time.monotonic()
                if recovery:
                    candidates = refine_centers(candidates, intensity)
                matches = self.match_candidates(
                    candidates, recovery, skip_pattern_ids=selected
                )
                # Keep the first valid threshold result for an identity. A later
                # threshold can split one bright LED into multiple contours and
                # create duplicate correspondences for the same already accepted
                # pattern; that must not invalidate the earlier match.
                fresh_matches = [
                    value for value in matches
                    if value["pattern_id"] not in selected
                ]
                stages["match_ms"] += (time.monotonic()-stage_start)*1000
                accepted, conflicts = reject_conflicts(fresh_matches)
                ambiguous.update(conflicts)
                for value in accepted:
                    value["threshold"] = threshold
                    # Preserve first valid baseline result, as in U+G.
                    selected.setdefault(value["pattern_id"], value)
                if selected and stages["first_match_ms"] is None:
                    stages["first_match_ms"] = (time.monotonic()-start)*1000
                if finished():
                    stages["exit_reason"] = "tracked_ids" if expected is not None else "all_ids"
                    break
            if finished():
                break
        final, conflicts = reject_conflicts(list(selected.values()))
        ambiguous.update(conflicts)
        # Conservative across thresholds: never resolve identity by iteration order.
        final = [value for value in final if value["pattern_id"] not in ambiguous]
        after = self.extractor.stats() if hasattr(self.extractor, "stats") else {}
        accelerator = {key: after.get(key, 0)-before.get(key, 0)
                       for key in set(before) | set(after)}
        return dict(schema="dohc.infrared.v1", detections=final,
                    method=method,
                    ambiguous_pattern_ids=sorted(ambiguous), backend=self.extractor.name,
                    accelerator=accelerator, stages=stages,
                    image_size=list(self.size), processing_ms=(time.monotonic()-start)*1000)

    def detect(self, image, timestamp=None):
        self.validate_image(image)
        intensity = cv2.min(image[:, :, 0], image[:, :, 2])
        color = cv2.bitwise_and(intensity,
            cv2.compare(cv2.subtract(intensity, image[:, :, 1]), 25, cv2.CMP_GT))
        green = image[:, :, 1].astype(np.float32)
        strength = intensity.astype(np.float32)
        weak_mask = (strength-green > np.maximum(5, .05*(strength+green+1))).astype(np.uint8)
        weak_mask = cv2.dilate(weak_mask, np.ones((3, 3), np.uint8)).astype(bool)
        weak_mask &= strength-cv2.blur(strength, (15, 15)) > 3
        weak = np.where(weak_mask, intensity, 0).astype(np.uint8)
        branches = ((color, [160, 180, 200, 220, 240] + list(range(155, 39, -5)), False),
                    (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), [160, 180, 200, 220, 240], False),
                    (weak, [160, 180, 200, 220, 240] + list(range(155, 9, -5)), True))
        return self._detect_branches(image, branches, intensity, "baseline", timestamp)


def main():
    import argparse
    from .methods import METHODS, create_detector
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "gpu-npu"))
    parser.add_argument("--npu-model", type=Path)
    parser.add_argument("--method", choices=METHODS, default="baseline")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Cannot decode image")
    detector = create_detector(args.method, json.loads(args.calibration.read_text()),
                               json.loads(args.patterns.read_text()), args.backend, args.npu_model)
    try:
        print(json.dumps(detector.detect(image), ensure_ascii=False, allow_nan=False))
    finally:
        detector.close()


if __name__ == "__main__":
    main()
