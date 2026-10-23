"""Completion-based replay/live-log benchmark; all timestamps use this host's clock."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from .runtime import InfraredWorker, clock_id
from .methods import ACCELERATED_METHODS, METHODS


def summarize(records, t0, t1, expected=None, method="baseline", max_age=5.):
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0:
        raise ValueError("Invalid monotonic window")
    duration = t1-t0
    seen, rows = set(), []
    for row in records:
        if "source_sequence" not in row or not t0 <= row["completed_monotonic"] < t1:
            continue
        key = (row["run_id"], row["source_sequence"])
        if key in seen:
            raise ValueError("Duplicate completion: %s" % (key,))
        seen.add(key)
        rows.append(row)
    done = [r for r in rows if r.get("state") == "ready"]
    correct, usable, estimates, orientations, wrong, stale, hardware_bad = [], [], [], [], 0, 0, 0
    for row in done:
        identities = [d["pattern_id"] for d in row.get("detections", [])]
        source = row.get("source_monotonic") if method == "gen2" else row["submitted_monotonic"]
        age = math.inf if source is None else row["completed_monotonic"]-source
        fresh = 0 <= age <= max_age and row.get("fresh", True)
        stale += int(not fresh)
        if expected is not None:
            wrong += int(bool(set(identities)-set(expected)) or len(identities) != len(set(identities)))
            if fresh and sorted(identities) == sorted(expected) and not row.get("ambiguous_pattern_ids"):
                correct.append(row)
                if all(d.get("quality", {}).get("pose_usable", not d.get("pose_ambiguous", False))
                       for d in row.get("detections", [])):
                    usable.append(row)
                if all(d.get('quality', {}).get('pose_available', False)
                       for d in row.get('detections', [])):
                    estimates.append(row)
                if identities and all(d.get('quality', {}).get('pose_usable', False) and
                        d.get('quality', {}).get('orientation_supported', False)
                        for d in row.get('detections', [])):
                    orientations.append(row)
        evidence = row.get("accelerator", {})
        hardware_bad += int(method in ACCELERATED_METHODS and (
            evidence.get("gpu_calls", 0) < 1 or evidence.get("npu_calls", 0) < 1
            or evidence.get("cpu_fallback", 0) != 0))

    def latency(values):
        values = sorted(values)
        return dict(n=len(values), p50_ms=statistics.median(values) if values else None,
                    p95_ms=values[math.ceil(.95*len(values))-1] if values else None,
                    max_ms=max(values) if values else None)

    def max_gap(selected):
        times = [t0] + sorted(r["completed_monotonic"] for r in selected) + [t1]
        return max(b-a for a, b in zip(times, times[1:]))

    bins = [dict(second=i, completed=0, correct=0) for i in range(math.ceil(duration))]
    for row in done:
        bins[int(row["completed_monotonic"]-t0)]["completed"] += 1
    for row in correct:
        bins[int(row["completed_monotonic"]-t0)]["correct"] += 1
    return dict(method=method, t0=t0, t1=t1, duration_seconds=duration,
                completed=len(done), errors=len(rows)-len(done),
                processed_fps=len(done)/duration,
                correct=len(correct) if expected is not None else None,
                correct_fps=len(correct)/duration if expected is not None else None,
                identity_success_rate=len(correct)/len(done) if done and expected is not None else None,
                usable_pose_frames=len(usable) if expected is not None else None,
                effective_pose_hz=len(usable)/duration if expected is not None else None,
                estimated_pose_frames=len(estimates) if expected is not None else None,
                effective_estimated_pose_hz=len(estimates)/duration if expected is not None else None,
                first_estimated_pose_seconds=(min(r['completed_monotonic'] for r in estimates)-t0) if estimates else None,
                max_no_estimated_pose_seconds=max_gap(estimates) if expected is not None else None,
                first_usable_pose_seconds=(min(r["completed_monotonic"] for r in usable)-t0) if usable else None,
                max_no_usable_pose_seconds=max_gap(usable) if expected is not None else None,
                orientation_supported_frames=len(orientations) if expected is not None else None,
                effective_orientation_supported_hz=len(orientations)/duration if expected is not None else None,
                wrong_id_frames=wrong if expected is not None else None,
                stale_frames=stale, hardware_invalid_frames=hardware_bad,
                processing=latency([r["processing_ms"] for r in done]),
                submit_to_complete=latency([(r["completed_monotonic"]-r["submitted_monotonic"])*1000 for r in done]),
                max_no_completion_seconds=max_gap(done),
                max_no_correct_seconds=max_gap(correct) if expected is not None else None,
                per_second=bins, status="PASS" if done and len(rows) == len(done) and not hardware_bad else "FAILED")


class CompletionTail:
    def __init__(self, path, start_at_end=False):
        self.stream = Path(path).open("rb")
        self.discard_partial = False
        if start_at_end:
            self.stream.seek(0, 2)
            if self.stream.tell():
                self.stream.seek(-1, 2)
                self.discard_partial = self.stream.read(1) != b"\n"
        self.pending = b""

    def read(self):
        self.pending += self.stream.read()
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()
        if self.discard_partial and lines:
            lines.pop(0)
            self.discard_partial = False
        return [json.loads(line) for line in lines if line]

    def close(self):
        self.stream.close()


def measure(args, method, entries=None):
    output = args.output / method
    output.mkdir(parents=True, exist_ok=False)
    worker, tail = None, None
    records, reason, selected_run = [], None, None
    t0 = t1 = None
    sequence, waiting = 0, False
    stage = "warmup"
    warm_started = time.monotonic()
    warm_count = 0
    hashes = {}
    for name, path in (("calibration", args.calibration), ("patterns", args.patterns), ("model", args.npu_model)):
        if path:
            hashes[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    metadata = dict(method=method, scope="live_completion_log" if args.live_log else "replay_not_camera_fps",
                    clock_id=clock_id(), hashes=hashes, expected_ids=args.expected_id,
                    duration_seconds=args.duration, warmup_seconds=args.warmup,
                    timeout_seconds=args.timeout, max_age_seconds=args.max_age,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")})
    release = Path(__file__).resolve().parents[2] / "RELEASE.json"
    if release.is_file():
        metadata["release"] = json.loads(release.read_text())
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    try:
        if args.live_log:
            tail = CompletionTail(args.live_log, start_at_end=True)
        else:
            log = output / "worker.jsonl"
            log.touch(exist_ok=False)
            worker = InfraredWorker(sys.executable, args.calibration, args.patterns,
                                    "gpu-npu" if method in ACCELERATED_METHODS else "cpu", args.npu_model,
                                    method=method, output=output / "latest.json", log_output=log,
                                    timeout=args.timeout, max_age=args.max_age)
            tail = CompletionTail(log)
        with (output / "frames.jsonl").open("w", encoding="utf-8") as saved:
            while True:
                now = time.monotonic()
                if stage == "warmup" and now-warm_started > args.timeout:
                    raise TimeoutError("Warmup timeout")
                if t1 is not None and now >= t1+args.timeout:
                    raise TimeoutError("Completion drain timeout; log may be incomplete")
                if worker is not None and not waiting:
                    entry = entries[sequence % len(entries)]
                    data = Path(entry).read_bytes()
                    sequence += 1
                    waiting = worker.submit(data, sequence, source_monotonic=time.monotonic(),
                        recording_started_monotonic=t0 if method == "gen2" else None,
                        source_time_kind="replay_frame_available")
                for row in tail.read():
                    records.append(row)
                    saved.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                    saved.flush()
                    if row.get("clock_id") != clock_id():
                        raise ValueError("Log clock differs from this boot; run sampler on DOHC")
                    if row.get("method") != method:
                        raise ValueError("Live method differs from requested profile")
                    if row.get("state") != "ready":
                        raise RuntimeError(row.get("error", "Detector stopped"))
                    if "source_sequence" not in row:
                        continue
                    waiting = False
                    if selected_run is None:
                        selected_run = row["run_id"]
                    if row["run_id"] != selected_run:
                        raise ValueError("Detector restarted during measurement")
                    if stage == "warmup":
                        warm_count += 1
                        if warm_count >= 3 and now-warm_started >= args.warmup:
                            t0 = time.monotonic()
                            t1 = t0+args.duration
                            stage = "measuring"
                    elif row["completed_monotonic"] >= t1:
                        stage = "complete"
                if stage == "complete":
                    break
                if worker is not None and worker.snapshot().get("state") == "error":
                    raise RuntimeError(worker.snapshot().get("error", "Worker failed"))
                time.sleep(.01)
    except Exception as exc:
        reason = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if worker is not None:
            worker.close()
        if tail is not None:
            tail.close()
    try:
        result = summarize(records, t0, t1, args.expected_id, method, args.max_age) if t0 is not None else dict(method=method, status="FAILED")
    except Exception as exc:
        result, reason = dict(method=method, status="FAILED"), str(exc)
    if reason:
        result.update(status="FAILED", reason=reason)
    result.update(scope=metadata["scope"], run_id=selected_run, warmup_completed=warm_count,
                  clock_id=clock_id(), drain_complete=stage == "complete")
    metadata.update(run_id=selected_run, t0=t0, t1=t1, status=result["status"],
                    drain_complete=stage == "complete")
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS+("all",), required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--manifest", type=Path, help='Ordered JSON array of {"image": "frame.png"} entries')
    inputs.add_argument("--live-log", type=Path, help="Completion JSONL on the same DOHC boot")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--patterns", type=Path)
    parser.add_argument("--npu-model", type=Path)
    parser.add_argument("--expected-id", action="append", help="Exact registry ID, repeat for multiple visible targets")
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--warmup", type=float, default=5.)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--max-age", type=float, default=5.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.duration, args.timeout, args.max_age)) or not math.isfinite(args.warmup) or args.warmup < 0:
        parser.error("Invalid duration, timeout, warmup or max-age")
    if args.live_log and args.method == "all":
        parser.error("Live measurement samples one active service method at a time")
    if not args.live_log and (not args.calibration or not args.patterns):
        parser.error("Replay requires explicit calibration and pattern registry")
    if args.expected_id and len(args.expected_id) != len(set(args.expected_id)):
        parser.error("Expected IDs must be unique")
    entries = None
    if args.image:
        entries = [args.image.resolve()]
    elif args.manifest:
        rows = json.loads(args.manifest.read_text())
        if not isinstance(rows, list) or not rows:
            parser.error("Manifest must be a nonempty ordered array")
        if any(not isinstance(row, dict) or not isinstance(row.get("image"), str)
               or not row["image"] for row in rows):
            parser.error("Each manifest entry requires a nonempty image path")
        entries = [(args.manifest.parent / row["image"]).resolve() for row in rows]
    if entries and any(not p.is_file() for p in entries):
        parser.error("Missing input image")
    if entries and any(str(p).startswith("/mnt/Datasets/") for p in entries):
        parser.error("Copy NAS data locally and verify it before replay")
    if args.patterns and args.expected_id:
        registry = {row["id"] for row in json.loads(args.patterns.read_text())}
        if not set(args.expected_id) <= registry:
            parser.error("Expected ID is absent from registry")
    methods = METHODS if args.method == "all" else (args.method,)
    results = [measure(args, method, entries) for method in methods]
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
