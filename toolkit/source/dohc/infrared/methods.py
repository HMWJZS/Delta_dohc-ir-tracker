"""Three online experiments sharing the same identities, calibration and gates."""
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from .detector import Candidate, CPUExtractor, MultiTracker, reject_conflicts

METHODS = ("baseline", "psf", "temporal_roi", "sparse_tiles",
           "gray_global_gpu_npu", "gray_adaptive_gpu_npu", "binary_morph_gpu_npu",
           "baseline_gpu_npu", "psf_gpu_npu", "temporal_roi_gpu_npu",
           "sparse_tiles_gpu_npu")
ACCELERATED_METHODS = METHODS[4:]
LEGACY_METHODS = METHODS
METHODS += ("gen2",)


def fit_spot(patch, initial):
    """Frozen P experiment: elliptical Gaussian + sloped background, <=1 px shift.

Ported from 20260920_far_precision_three/methods.py; this is an additional
experiment, not part of the adopted G baseline. Saturated pixels are excluded.
"""
    yy, xx = np.indices(patch.shape, dtype=float)
    xx -= initial[0]
    yy -= initial[1]
    usable = patch < 250
    background = float(np.percentile(patch, 20))
    amplitude = float(patch.max()-background)
    if usable.sum() < 16 or amplitude < 8:
        return np.asarray(initial), False
    radius = (min(patch.shape)-1)/2

    def residual(value):
        a, x, y, sx, sy, b, bx, by = value
        model = b+bx*xx+by*yy+a*np.exp(-.5*((xx-x)**2/sx**2+(yy-y)**2/sy**2))
        return (model-patch)[usable]

    fit = least_squares(residual, [amplitude, 0, 0, .8, .8, background, 0, 0],
        bounds=([0, -1, -1, .45, .45, 0, -50, -50],
                [510, 1, 1, radius, radius, 255, 50, 50]),
        loss="soft_l1", f_scale=3., max_nfev=40)
    accepted = (fit.success and np.isfinite(fit.x).all() and fit.x[0] >= 8
        and np.linalg.norm(fit.x[1:3]) <= 1.
        and np.sqrt(np.mean(residual(fit.x)**2)) <= max(3., .3*fit.x[0])
        and max(fit.x[3:5]) < radius*.98)
    return np.asarray(initial)+(fit.x[1:3] if accepted else 0), bool(accepted)


class PSFTracker(MultiTracker):
    """Accuracy experiment: fit only identified LEDs, then rerun gated G pose."""

    def detect(self, image, timestamp=None):
        started = time.monotonic()
        result = super().detect(image, timestamp=timestamp)
        patterns = {p["id"]: p for p in self.patterns}
        for detection in result["detections"]:
            points = np.asarray(detection["points_raw"])
            distance = np.linalg.norm(points[:, None]-points[None], axis=2)
            np.fill_diagonal(distance, np.inf)
            refined = points.copy()
            count = 0
            for i, (x, y) in enumerate(points):
                radius = int(np.clip(np.floor(.45*distance[i].min()), 2, 5))
                px, py = np.rint([x, y]).astype(int)
                if px-radius < 0 or py-radius < 0 or px+radius >= image.shape[1] or py+radius >= image.shape[0]:
                    continue
                region = image[py-radius:py+radius+1, px-radius:px+radius+1]
                patch = np.minimum(region[:, :, 0], region[:, :, 2]).astype(float)
                center, valid = fit_spot(patch, [x-px+radius, y-py+radius])
                if valid:
                    refined[i] = center + [px-radius, py-radius]
                    count += 1
            pose = self.pose(patterns[detection["pattern_id"]], refined,
                             detection["source"] == "recovery") if count else None
            detection["psf"] = dict(fitted_leds=count, applied=pose is not None)
            if pose is not None:
                pose["pose_ambiguous"] |= detection["pose_ambiguous"]
                detection.update(pose, points_raw=refined.tolist())
        result["detections"], conflicts = reject_conflicts(result["detections"])
        result["ambiguous_pattern_ids"] = sorted(set(result["ambiguous_pattern_ids"]+conflicts))
        result.update(method="psf", processing_ms=(time.monotonic()-started)*1000)
        return result


def merge_rectangles(rectangles):
    """Merge overlapping/touching native-pixel rectangles; at most five targets."""
    result = []
    for rectangle in rectangles:
        box = list(rectangle)
        index = 0
        while index < len(result):
            old = result[index]
            if box[0] <= old[2] and old[0] <= box[2] and box[1] <= old[3] and old[1] <= box[3]:
                box = [min(box[0], old[0]), min(box[1], old[1]), max(box[2], old[2]), max(box[3], old[3])]
                result.pop(index)
                index = 0
            else:
                index += 1
        result.append(box)
    return result


class WindowExtractor:
    name = "cpu_temporal_roi"

    def __init__(self, rectangles):
        self.rectangles = merge_rectangles(rectangles)
        self.edge_fallback = False

    def prepare(self, gray):
        self.gray = gray
        self.full = None
        self.patches = []
        height, width = gray.shape
        for x0, y0, x1, y1 in self.rectangles:
            left, top = max(0, x0-2), max(0, y0-2)
            right, bottom = min(width, x1+2), min(height, y1+2)
            patch = CPUExtractor()
            patch.prepare(gray[top:bottom, left:right])
            core = patch.smooth[y0-top:y1-top, x0-left:x1-left].copy()
            # Interior pixels have two-pixel Gaussian context. A component on
            # a crop edge must trigger a full-frame retry, never become a cut LED.
            patch.gray = gray[y0:y1, x0:x1]
            patch.weights = patch.gray
            patch.smooth = core
            edges = []
            if y0 > 0: edges.append(core[0])
            if y1 < height: edges.append(core[-1])
            if x0 > 0: edges.append(core[:, 0])
            if x1 < width: edges.append(core[:, -1])
            self.patches.append((x0, y0, patch, max((int(e.max()) for e in edges), default=0)))

    def extract(self, threshold, min_area=2):
        if any(edge > threshold for _, _, _, edge in self.patches):
            self.edge_fallback = True
            if self.full is None:
                self.full = CPUExtractor()
                self.full.prepare(self.gray)
            return self.full.extract(threshold, min_area)
        found = []
        for x, y, patch, _ in self.patches:
            found.extend(Candidate(c.center+np.float32([x, y]), c.peak, c.core_area)
                         for c in patch.extract(threshold, min_area))
            if len(found) > 2048:
                return found[:2049]
        found.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
        return found


class TemporalROITracker(MultiTracker):
    """Speed experiment: native-pixel ROIs, periodic full scan, immediate lost retry."""

    def __init__(self, *args, refresh_frames=10, refresh_seconds=.5, padding=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if (type(refresh_frames) is not int or refresh_frames < 1
                or not np.isfinite([refresh_seconds, padding]).all()
                or refresh_seconds <= 0 or padding < 0):
            raise ValueError("Invalid ROI refresh/padding limits")
        self.refresh_frames, self.refresh_seconds, self.padding = refresh_frames, refresh_seconds, padding
        self.previous = []
        self.previous_time = None
        self.full_time = None
        self.since_full = 0

    def search_regions(self, stamp):
        age = None if self.previous_time is None else stamp-self.previous_time
        # A slow full scan may return an old observation. Permit one bounded,
        # expanded ROI verification on fresh pixels before another full scan.
        bootstrap = (self.since_full == 0 and self.full_time is not None
                     and stamp >= self.full_time
                     and stamp-self.full_time < self.refresh_seconds)
        full = (not self.previous or age is None or age <= 0 or age > 5.
                or (age > self.refresh_seconds and not bootstrap)
                or self.full_time is None or stamp-self.full_time >= self.refresh_seconds
                or self.since_full >= self.refresh_frames-1)
        rectangles = []
        if not full:
            for detection in self.previous:
                points = np.asarray(detection["points_raw"])
                margin = max(12., self.padding*np.linalg.norm(np.ptp(points, axis=0)))
                margin += min(480., max(0., age-self.refresh_seconds)*120.)
                lo = np.maximum(np.floor(points.min(0)-margin), 0).astype(int)
                hi = np.minimum(np.ceil(points.max(0)+margin+1), self.size).astype(int)
                rectangles.append([int(value) for value in (*lo, *hi)])
        return full, rectangles

    def detect(self, image, timestamp=None):
        started = time.monotonic()
        stamp = started if timestamp is None else float(timestamp)
        if not np.isfinite(stamp):
            raise ValueError("Frame timestamp must be finite")
        full, rectangles = self.search_regions(stamp)
        original = self.extractor
        mode = "full"
        try:
            if not full:
                self.extractor = WindowExtractor(rectangles)
                self.tracking_pattern_ids = {d["pattern_id"] for d in self.previous}
                result = super().detect(image, timestamp=stamp)
                mode = "roi_edge_full" if self.extractor.edge_fallback else "roi"
                expected = {d["pattern_id"] for d in self.previous}
                full = bool(result["ambiguous_pattern_ids"]) or expected != {d["pattern_id"] for d in result["detections"]}
                if full:
                    mode = "lost_full_retry"
            if full:
                self.extractor = original
                self.tracking_pattern_ids = None
                result = super().detect(image, timestamp=stamp)
                self.full_time, self.since_full = stamp+(time.monotonic()-started), 0
            else:
                self.since_full += 1
        finally:
            self.extractor = original
            self.tracking_pattern_ids = None
        self.previous, self.previous_time = result["detections"], stamp
        result.update(method="temporal_roi", search_mode=mode, roi_count=len(rectangles),
                      processing_ms=(time.monotonic()-started)*1000)
        return result


class SparseTileExtractor:
    """Stateless coarse tile proposals with exact native threshold components."""
    name = "cpu_sparse_tiles"

    def prepare(self, gray):
        self.full = CPUExtractor()
        self.full.prepare(gray)
        height, width = gray.shape
        smooth = self.full.smooth
        padded = np.pad(smooth, ((0, (-height) % 16), (0, (-width) % 16)))
        tiles = padded.reshape(padded.shape[0]//16, 16, padded.shape[1]//16, 16).max(axis=(1, 3))
        count, labels, stats, _ = cv2.connectedComponentsWithStats((tiles > 10).astype(np.uint8), connectivity=8)
        area = sum(int(row[2])*int(row[3])*256 for row in stats[1:])
        self.patches = []
        # ponytail: bounded patch list; dense scenes fall back to the exact full scan.
        self.dense = count > 129 or area > gray.size*.5
        if self.dense:
            return
        for label in range(1, count):
            tx, ty, tw, th, _ = map(int, stats[label])
            x, y = tx*16, ty*16
            right, bottom = min(width, (tx+tw)*16), min(height, (ty+th)*16)
            mask = np.repeat(np.repeat(labels[ty:ty+th, tx:tx+tw] == label, 16, axis=0), 16, axis=1)
            patch = CPUExtractor()
            patch.gray = gray[y:bottom, x:right]
            patch.weights = patch.gray
            # Mask ownership even when bounding boxes of coarse components overlap.
            patch.smooth = np.where(mask[:bottom-y, :right-x], smooth[y:bottom, x:right], 0)
            self.patches.append((x, y, patch))

    def extract(self, threshold, min_area=2):
        if threshold < 10:
            raise ValueError("Sparse tile support requires threshold >=10")
        if self.dense:
            return self.full.extract(threshold, min_area)
        found = []
        for x, y, patch in self.patches:
            found.extend(Candidate(c.center+np.float32([x, y]), c.peak, c.core_area)
                         for c in patch.extract(threshold, min_area))
            if len(found) > 2048:
                return found[:2049]
        # Deterministic spatial tie order; geometric matching is permutation independent.
        found.sort(key=lambda c: (float(c.center[1]), float(c.center[0])))
        found.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
        return found


class SparseTileTracker(MultiTracker):
    def __init__(self, calibration, patterns, extractor=None):
        if extractor is not None and not isinstance(extractor, CPUExtractor):
            raise ValueError("sparse_tiles currently uses CPU patches; select --backend cpu")
        super().__init__(calibration, patterns, SparseTileExtractor())

    def detect(self, image, timestamp=None):
        result = super().detect(image, timestamp=timestamp)
        result["method"] = "sparse_tiles"
        return result


def _require_accelerators(extractor, method):
    if not getattr(extractor, "is_gpu", False) or not getattr(extractor, "is_npu", False):
        raise ValueError("%s requires OpenCL GPU and RKNN NPU; no CPU fallback" % method)


class GrayTracker(MultiTracker):
    def __init__(self, calibration, patterns, extractor, mode):
        _require_accelerators(extractor, mode)
        super().__init__(calibration, patterns, extractor)
        self.mode = mode

    def detect(self, image, timestamp=None):
        self.validate_image(image)
        gray, signal = self.extractor.preprocess(image, self.mode)
        if self.mode == "gray_global":
            thresholds, recovery, weights = [160, 180, 200, 220, 240] + list(range(155, 9, -5)), False, None
        elif self.mode == "gray_adaptive":
            thresholds, recovery, weights = [10, 12, 16, 20, 24, 32], True, gray
        else:
            thresholds, recovery, weights = [10], False, gray
        # Binary segmentation must not be blurred back into a grayscale mask.
        original = self.extractor
        if self.mode == "binary_morph":
            self.extractor = BinaryExtractor(original)
        try:
            return self._detect_branches(image, ((signal, thresholds, recovery, weights),),
                                         gray, self.mode + "_gpu_npu", timestamp,
                                         min_area=1 if self.mode == "binary_morph" else None)
        finally:
            self.extractor = original


class BinaryExtractor:
    def __init__(self, gpu):
        self.gpu, self.name = gpu, gpu.name

    def prepare(self, signal, weights):
        self.gpu.prepare(signal, weights, binary=True)

    def extract(self, threshold, min_area=2):
        return self.gpu.extract(threshold, min_area)

    def stats(self):
        return self.gpu.stats()


def proposal_rectangles(maxima, size):
    mask = np.asarray(maxima > 10, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Bound coarse-map work before merging; dense data uses the full GPU.
    if count > 129:
        return None
    rectangles = [[max(0, int(x)*16-16), max(0, int(y)*16-16),
                   min(size[0], int(x+w)*16+16), min(size[1], int(y+h)*16+16)]
                  for x, y, w, h, _ in stats[1:]]
    rectangles = merge_rectangles(rectangles)
    if sum((r[2]-r[0])*(r[3]-r[1]) for r in rectangles) > size[0]*size[1]*.65:
        return None
    return rectangles


class GPUSpatialExtractor:
    """NPU proposals select nonoverlapping native-pixel GPU patches per branch."""
    name = "rk3588_npu+mali_opencl_regions"

    def __init__(self, gpu, size, history=(), full=False):
        self.gpu, self.size, self.history, self.full = gpu, size, history, full
        self.edge_fallback = False
        self.region_count = 0
        self.region_pixels = 0

    def prepare(self, gray):
        self.gray = gray
        maxima = self.gpu.npu.maxima(gray)
        rectangles = proposal_rectangles(maxima, self.size)
        self.rectangles = None if self.full or rectangles is None else merge_rectangles(
            rectangles + list(self.history))
        active = maxima > 10
        padded = np.pad(active, 1)
        self.tiles = np.zeros_like(active, dtype=np.uint8)
        for y in range(3):
            for x in range(3):
                self.tiles |= padded[y:y+active.shape[0], x:x+active.shape[1]]
        if self.rectangles == []:
            # A full masked scan makes a blank result depend on the NPU output.
            self.rectangles = None
        self.region_count = max(self.region_count, len(self.rectangles or []))

    def extract(self, threshold, min_area=2):
        if threshold < 10:
            raise ValueError("NPU spatial proposals require threshold >=10")
        if self.rectangles is None:
            self.gpu.prepare(self.gray, use_npu=False, tile_map=self.tiles)
            self.region_pixels += self.gray.size
            return self.gpu.extract(threshold, min_area)
        found = []
        for x0, y0, x1, y1 in self.rectangles:
            left, top = max(0, x0-2), max(0, y0-2)
            right, bottom = min(self.size[0], x1+2), min(self.size[1], y1+2)
            sides = int(x0 > 0) + 2*int(y0 > 0) + 4*int(x1 < self.size[0]) + 8*int(y1 < self.size[1])
            self.gpu.prepare(self.gray[top:bottom, left:right], use_npu=False,
                             core=(x0-left, y0-top, x1-left, y1-top, sides))
            self.region_pixels += (right-left)*(bottom-top)
            if self.gpu.edge_peak > threshold:
                self.edge_fallback = True
                self.gpu.prepare(self.gray, use_npu=False, tile_map=self.tiles)
                self.region_pixels += self.gray.size
                return self.gpu.extract(threshold, min_area)
            found.extend(Candidate(c.center+np.float32([left, top]), c.peak, c.core_area)
                         for c in self.gpu.extract(threshold, min_area))
            if len(found) > 2048:
                return found[:2049]
        found.sort(key=lambda c: (float(c.center[1]), float(c.center[0])))
        found.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
        return found

    def stats(self):
        return self.gpu.stats()


class AcceleratedTemporalROITracker(TemporalROITracker):
    def detect(self, image, timestamp=None):
        self.validate_image(image)
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(stamp):
            raise ValueError("Frame timestamp must be finite")
        started = time.monotonic()
        full, rectangles = self.search_regions(stamp)
        original = self.extractor
        spatial = GPUSpatialExtractor(original, self.size, rectangles, full)
        self.extractor = spatial
        mode = "full_gpu_npu" if full else "roi_gpu_npu"
        self.tracking_pattern_ids = None if full else {d["pattern_id"] for d in self.previous}
        try:
            result = MultiTracker.detect(self, image, stamp)
            if not full and (result["ambiguous_pattern_ids"] or
                    {d["pattern_id"] for d in self.previous} !=
                    {d["pattern_id"] for d in result["detections"]} or spatial.edge_fallback):
                spatial.full = True
                self.tracking_pattern_ids = None
                result = MultiTracker.detect(self, image, stamp)
                mode, full = "lost_full_gpu_npu", True
        finally:
            self.extractor = original
            self.tracking_pattern_ids = None
        if full:
            self.full_time, self.since_full = stamp+(time.monotonic()-started), 0
        else:
            self.since_full += 1
        self.previous, self.previous_time = result["detections"], stamp
        result.update(search_mode=mode, roi_count=spatial.region_count,
                      gpu_region_pixels=spatial.region_pixels)
        return result


class AcceleratedSparseTracker(MultiTracker):
    def detect(self, image, timestamp=None):
        original = self.extractor
        spatial = GPUSpatialExtractor(original, self.size)
        self.extractor = spatial
        try:
            result = super().detect(image, timestamp)
        finally:
            self.extractor = original
        result.update(search_mode="sparse_gpu_npu", roi_count=spatial.region_count,
                      gpu_region_pixels=spatial.region_pixels)
        return result


class ProfiledDetector:
    """One frame boundary for timing and mandatory accelerator completion proof."""
    def __init__(self, tracker, method):
        self.tracker, self.method = tracker, method

    def __getattr__(self, name):
        return getattr(self.tracker, name)

    def detect(self, image, timestamp=None, recording_started_monotonic=None):
        self.tracker.validate_image(image)
        if timestamp is not None and not np.isfinite(float(timestamp)):
            raise ValueError("Frame timestamp must be finite")
        extractor = self.tracker.extractor
        try:
            before = extractor.stats() if hasattr(extractor, "stats") else {}
            started = time.monotonic()
            if self.method == 'gen2':
                result = self.tracker.detect(image, timestamp,
                                             recording_started_monotonic=recording_started_monotonic)
            else:
                result = self.tracker.detect(image, timestamp)
            after = extractor.stats() if hasattr(extractor, "stats") else {}
        except Exception:
            if self.method == 'gen2':
                self.tracker.reset()
            raise
        evidence = {key: after.get(key, 0)-before.get(key, 0) for key in set(before) | set(after)}
        if self.method in ACCELERATED_METHODS and (
                evidence.get("gpu_calls", 0) < 1 or evidence.get("npu_calls", 0) < 1
                or evidence.get("cpu_fallback", 0)):
            raise RuntimeError("GPU+NPU profile completed without both accelerators")
        result.update(method=self.method, accelerator=evidence,
                      processing_ms=(time.monotonic()-started)*1000)
        if self.method == 'gen2' and getattr(extractor, 'is_gpu', False):
            result['gpu'] = dict(device=extractor.device,
                                 full_components='gpu' if self.tracker.gpu_components == 'auto' else 'cpu',
                                 roi_components='cpu', weak_branch='cpu')
        return result

    def close(self):
        extractor = self.tracker.extractor
        try:
            npu = getattr(extractor, "npu", None)
            if npu is not None:
                npu.close()
        finally:
            if hasattr(extractor, 'close'):
                extractor.close()


def make_detector(method, calibration, patterns, extractor=None):
    if method not in METHODS:
        raise ValueError("Unknown infrared method: %s" % method)
    if method in ACCELERATED_METHODS:
        _require_accelerators(extractor, method)
    if method == "gen2":
        from .gen2 import Gen2Tracker
        tracker = Gen2Tracker(calibration, patterns, extractor)
    elif method in METHODS[4:7]:
        tracker = GrayTracker(calibration, patterns, extractor, method[:-8])
    else:
        choices = dict(baseline=MultiTracker, psf=PSFTracker,
                       temporal_roi=TemporalROITracker, sparse_tiles=SparseTileTracker,
                       baseline_gpu_npu=MultiTracker, psf_gpu_npu=PSFTracker,
                       temporal_roi_gpu_npu=AcceleratedTemporalROITracker,
                       sparse_tiles_gpu_npu=AcceleratedSparseTracker)
        tracker = choices[method](calibration, patterns, extractor)
    return ProfiledDetector(tracker, method)


def create_detector(method, calibration, patterns, backend=None, model=None):
    """Production construction shared by CLI, runtime and field tests."""
    backend = backend or ("gpu-npu" if method in ACCELERATED_METHODS else "cpu")
    if backend not in ("cpu", "gpu", "gpu-npu"):
        raise ValueError("Unknown backend: %s" % backend)
    if method in ACCELERATED_METHODS and backend != "gpu-npu":
        raise ValueError("This profile requires --backend gpu-npu")
    if method == "gen2" and backend == "gpu-npu":
        raise ValueError("Gen2 supports --backend cpu or gpu")
    if backend == "cpu":
        return make_detector(method, calibration, patterns)
    from .opencl_extract import OpenCLExtractor
    npu = extractor = None
    try:
        if backend == "gpu-npu":
            if not model:
                raise ValueError("GPU+NPU requires an RKNN model")
            from .npu_tiles import NPUTiles
            npu = NPUTiles(model)
        extractor = OpenCLExtractor(npu)
        return make_detector(method, calibration, patterns, extractor)
    except Exception:
        try:
            if npu is not None:
                npu.close()
        finally:
            if extractor is not None:
                extractor.close()
        raise
