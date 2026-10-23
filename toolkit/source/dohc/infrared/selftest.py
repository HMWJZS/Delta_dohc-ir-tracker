"""Board-only accelerator equivalence and eleven-profile synthetic smoke test."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .detector import CPUExtractor, MultiTracker
from .methods import ACCELERATED_METHODS, LEGACY_METHODS, METHODS, make_detector
from .npu_tiles import NPUTiles
from .opencl_extract import OpenCLExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-model", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True,
                        help="confirmed field registry; do not use the synthetic test fixture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    root = Path(__file__).parent
    calibration = json.loads((root / "reference_940nm_intrinsics.json").read_text())
    patterns = json.loads(args.patterns.read_text())
    npu, report = None, dict(scope="synthetic_only", status="FAILED", profiles={})
    try:
        npu = NPUTiles(args.npu_model)
        gpu = OpenCLExtractor(npu)
        report["gpu_device"] = gpu.device
        image = np.zeros((2160, 3840, 3), np.uint8)
        reference = MultiTracker(calibration, patterns)
        for i, pattern in enumerate(reference.patterns):
            points = cv2.fisheye.projectPoints(pattern["objects"][:, None],
                np.array([.25, .15, .1]), np.array([(i-2)*.2, 0., .5]),
                reference.K, reference.D)[0][:, 0]
            for point in points:
                cv2.circle(image, tuple(np.rint(point).astype(int)), 2, (240, 20, 240), -1)
        for gray in (image[:, :, 0].copy(), np.zeros(image.shape[:2], np.uint8),
                     np.random.default_rng(71).integers(0, 256, image.shape[:2], dtype=np.uint8)):
            np.testing.assert_array_equal(npu.maxima(gray), gray.reshape(135, 16, 240, 16).max(axis=(1, 3)))
        for mode in ("gray_global", "gray_adaptive", "binary_morph"):
            raw, signal = gpu.preprocess(image, mode)
            bgr = image.astype(np.int32)
            expected = ((1868*bgr[:, :, 0]+9617*bgr[:, :, 1]+4899*bgr[:, :, 2]+8192)>>14).astype(np.uint8)
            np.testing.assert_array_equal(raw, expected)
            if mode == "gray_adaptive":
                expected = cv2.subtract(raw, cv2.blur(raw, (15, 15)))
                expected[raw <= 5] = 0
            elif mode == "binary_morph":
                expected = cv2.morphologyEx((raw > 80).astype(np.uint8)*255, cv2.MORPH_CLOSE,
                                           np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
            np.testing.assert_array_equal(signal, expected)
        full = CPUExtractor()
        plane = image[:, :, 0].copy()
        full.prepare(plane)
        gpu.prepare(plane)
        key = lambda c: tuple(c.center)
        expected = sorted(full.extract(100), key=key)
        actual = sorted(gpu.extract(100), key=key)
        np.testing.assert_allclose([[*c.center, c.peak, c.core_area] for c in actual],
                                   [[*c.center, c.peak, c.core_area] for c in expected], atol=1e-5)
        expected_ids = {p["id"] for p in patterns}
        for method in LEGACY_METHODS:
            tracker = make_detector(method, calibration, patterns, gpu if method in ACCELERATED_METHODS else None)
            started = time.monotonic()
            first = tracker.detect(image, 1.)
            second = tracker.detect(image, 1.03)
            blank = tracker.detect(np.zeros_like(image), 1.06)
            if any({d["pattern_id"] for d in r["detections"]} != expected_ids for r in (first, second)) or blank["detections"]:
                raise AssertionError("Incorrect synthetic identity set: %s" % method)
            report["profiles"][method] = dict(first=first, second=second, blank=blank,
                                              wall_ms=(time.monotonic()-started)*1000)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if npu is not None:
            npu.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output), error=report.get("error"))))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
