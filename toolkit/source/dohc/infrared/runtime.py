"""Latest-frame inference bridge; acquisition stays independent of detection."""
import json
import math
from collections import deque
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time
import uuid

from .version import digest, module_identity
from .config import CONFIG
from .profiling import enabled as profiling_enabled

# Internal paired producer/consumer packet. External schema stays dohc.infrared.v1.
HEADER = struct.Struct("!QI")
RESULT_TIMING_HEADER = struct.Struct('!8sQQQQ')
RESULT_TIMING_MAGIC = b'IRTIME01'
MAX_PACKET = 32 * 1024 * 1024
MAX_METADATA = 4096
GEN2_MAX_AGE = CONFIG["measurement_max_age_seconds"]
STARTUP_SECONDS = CONFIG["startup_seconds"]


def send_result(connection, result, diagnostic=False):
    started = time.monotonic_ns() if diagnostic else None
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    if diagnostic:
        serialized = time.monotonic_ns()
        if len(payload)+RESULT_TIMING_HEADER.size > MAX_PACKET:
            raise ValueError('Infrared result exceeds packet budget')
        payload = RESULT_TIMING_HEADER.pack(RESULT_TIMING_MAGIC, result['sequence'],
            started, serialized, time.monotonic_ns())+payload
    connection.send_bytes(payload)


def clock_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def phase_at(now, recording_started):
    if recording_started is None:
        return "preview", CONFIG["transition_hz"]
    return ("startup", CONFIG["startup_hz"]) if now-recording_started < STARTUP_SECONDS else ("transition", CONFIG["transition_hz"])


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Frame metadata must be an object")
    for key in ("submitted_monotonic", "source_monotonic", "recording_started_monotonic"):
        value = metadata.get(key)
        if value is None and key != "submitted_monotonic":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid monotonic time: %s" % key)
    if metadata.get("source_monotonic") is not None and metadata['source_monotonic'] > metadata['submitted_monotonic']+.05:
        raise ValueError("Source time must use the submitter's MONOTONIC clock")
    if metadata.get("recording_started_monotonic") is not None and metadata['recording_started_monotonic'] > metadata['submitted_monotonic']+.05:
        raise ValueError("Recording start is in the future")


def encode_packet(sequence, metadata, frame):
    validate_metadata(metadata)
    raw = json.dumps(metadata, allow_nan=False, separators=(',', ':')).encode()
    if len(raw) > MAX_METADATA or not frame:
        raise ValueError("Invalid metadata/image size")
    packet = HEADER.pack(sequence, len(raw))+raw+bytes(frame)
    if len(packet) > MAX_PACKET:
        raise ValueError("Infrared input exceeds packet budget")
    return packet


def decode_packet(packet):
    if not HEADER.size < len(packet) <= MAX_PACKET:
        raise ValueError("Truncated/oversized infrared frame")
    sequence, length = HEADER.unpack(packet[:HEADER.size])
    if not 0 < length <= MAX_METADATA or HEADER.size+length >= len(packet):
        raise ValueError("Invalid frame metadata length")
    metadata = json.loads(packet[HEADER.size:HEADER.size+length])
    validate_metadata(metadata)
    return sequence, metadata, packet[HEADER.size+length:]


def fresh_result(result, now, max_age, gen2=False):
    """A result's source age never becomes smaller by publishing it again."""
    result = dict(result)
    source = result.get('source_monotonic' if gen2 else 'submitted_monotonic')
    age = None if source is None else now-source
    fresh = (age is not None and 0 <= age <= max_age and result.get('state') == 'ready')
    result.update(input_age_seconds=age, fresh=fresh)
    if gen2:
        result.update(measurement_age_seconds=age, source_clock='MONOTONIC',
                      source_time_kind=result.get('source_time_kind', 'application_frame_arrival'))
    for field in ('detections', 'observations'):
        values = result.get(field, [])
        result[field] = [dict(d, quality=dict(d.get('quality', {}), measurement_fresh=fresh,
            pose_available=fresh and d.get('quality', {}).get('pose_available', False),
            position_usable=fresh and d.get('quality', {}).get('position_usable', False),
            orientation_supported=fresh and d.get('quality', {}).get('orientation_supported', False),
            pose_usable=fresh and d.get('quality', {}).get('pose_usable', not d.get('pose_ambiguous', False))))
            for d in values] if fresh else []
    if gen2:
        result['quality'] = dict(result.get('quality', {}), measurement_fresh=fresh,
                                 state=result.get('quality', {}).get('state', 'SEARCHING') if fresh else 'EXPIRED')
    return result


def empty_metrics():
    return {phase: dict(completed=0, identity_frames=0, pose_frames=0, position_frames=0,
                        estimated_pose_frames=0, orientation_supported_frames=0,
                        first_orientation_supported_monotonic=None, last_orientation_supported_monotonic=None,
                        max_orientation_supported_gap_seconds=0.,
                        first_estimated_pose_monotonic=None, last_estimated_pose_monotonic=None,
                        max_estimated_pose_gap_seconds=0.,
                        first_position_monotonic=None, last_position_monotonic=None,
                        max_position_gap_seconds=0.,
                        first_pose_monotonic=None, last_pose_monotonic=None,
                        max_pose_gap_seconds=0., by_pattern={}) for phase in ('startup', 'transition')}


def count_completion(metric, result, phase_start):
    completed = result['completed_monotonic']
    metric['completed'] += 1
    detections = result['detections']
    metric['identity_frames'] += int(bool(detections))
    usable = [d for d in detections if d.get('quality', {}).get('pose_usable', False)]
    positions = [d for d in detections if d.get('quality', {}).get('position_usable', False)]
    estimates = [d for d in detections if d.get('quality', {}).get('pose_available', False)]
    orientations = [d for d in usable if d.get('quality', {}).get('orientation_supported', False)]
    for kind, values in [('pose', usable), ('position', positions), ('estimated_pose', estimates),
                         ('orientation_supported', orientations)]:
        metric[kind+'_frames'] += int(bool(values))
        if values:
            previous = metric['last_'+kind+'_monotonic']
            metric['max_'+kind+'_gap_seconds'] = max(metric['max_'+kind+'_gap_seconds'],
                completed-(phase_start if previous is None else previous))
            if metric['first_'+kind+'_monotonic'] is None:
                metric['first_'+kind+'_monotonic'] = completed
            metric['last_'+kind+'_monotonic'] = completed
    for detection in detections:
        target = metric['by_pattern'].setdefault(detection['pattern_id'],
            dict(identity_frames=0, pose_frames=0, position_frames=0, estimated_pose_frames=0, orientation_supported_frames=0))
        target['identity_frames'] += 1
        target['pose_frames'] += int(detection in usable)
        target['position_frames'] += int(detection in positions)
        target['estimated_pose_frames'] += int(detection in estimates)
        target['orientation_supported_frames'] += int(detection in orientations)


class LatestFrameFeeder:
    """Keep camera admission independent of preview, recording and inference work."""
    def __init__(self, camera, submit, on_error, profiling=False):
        self.camera, self.submit, self.on_error = camera, submit, on_error
        self.profiling = profiling
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='infrared-admission', daemon=True)
        self.thread.start()

    def run(self):
        sequence = None
        try:
            while not self.stop_event.is_set():
                sample = self.camera.wait_latest_sample(sequence, trace=self.profiling)
                if sample is None or self.stop_event.is_set():
                    continue
                sequence = sample['sequence']
                if self.profiling:
                    sample['runtime_timestamps']['ingress_submit_callback'] = time.monotonic()
                self.submit(sample)
        except Exception as exc:
            self.on_error(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError('Infrared admission thread did not stop')


class _CompletionLog:
    """One writer, two queued completions; full queues apply measured backpressure."""
    capacity = 2

    def __init__(self, write):
        self.write = write
        self.condition = threading.Condition()
        self.pending = deque()
        self.closing, self.error, self.active = False, None, None
        self.thread = threading.Thread(target=self.run, name='infrared-log', daemon=True)
        self.thread.start()

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Infrared log failed for sequence %s: %s' %
                                   (self.active[0].get('source_sequence'), self.error)) from self.error

    def put(self, result, frame, timings):
        requested = time.monotonic()
        with self.condition:
            if not self.condition.wait_for(
                    lambda: self.error is not None or self.closing or len(self.pending) < self.capacity,
                    timeout=5):
                raise TimeoutError('Infrared diagnostic log queue stalled')
            self.check()
            if self.closing:
                raise RuntimeError('Infrared diagnostic log is closed')
            queued = time.monotonic()
            timings.update(log_queue_requested_monotonic=requested,
                           log_enqueued_monotonic=queued,
                           logging_backpressure_ms=(queued-requested)*1000)
            self.pending.append((result, frame, timings))
            self.condition.notify_all()

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closing or self.pending)
                if not self.pending:
                    return
                self.active = self.pending.popleft()
                self.condition.notify_all()
            try:
                self.write(*self.active)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()
                return
            with self.condition:
                self.active = None

    def unwritten_sequences(self):
        with self.condition:
            jobs = ([self.active] if self.active is not None else [])+list(self.pending)
            return [job[0].get('source_sequence') for job in jobs]

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError('Infrared log drain timed out; unwritten sequences: %s' %
                               self.unwritten_sequences())
        self.check()


class InfraredWorker:
    def __init__(self, python, calibration, patterns, backend="gpu", model=None,
                 output="/run/dohc/infrared.json", timeout=60., max_age=None, method="baseline",
                 log_output=None, jpeg_decoder=None, profiling=None):
        self.jpeg_decoder = (jpeg_decoder or os.environ.get('DOHC_IR_TRACKER_JPEG_DECODER',
            'mpp_rga' if method == 'gen2' else 'cpu'))
        if self.jpeg_decoder not in ('cpu', 'mpp_rga'):
            raise ValueError('Unknown JPEG decoder: '+str(self.jpeg_decoder))
        if method != 'gen2' and self.jpeg_decoder != 'cpu':
            raise ValueError('MPP/RGA image sources require gen2')
        self.arguments = [python, "-m", __package__+".runtime", "--calibration", str(calibration),
                          "--patterns", str(patterns), "--backend", backend, "--method", method,
                          "--jpeg-decoder", self.jpeg_decoder]
        if model:
            self.arguments.extend(["--npu-model", str(model)])
        self.output = None if output is None else Path(output)
        self.temporary_output = None if self.output is None else self.output.with_suffix(".tmp")
        self.log_output = None if log_output is None else Path(log_output)
        self.profile_override = profiling
        self.profiling = (profiling_enabled() if profiling is None else profiling) and self.log_output is not None and method == 'gen2'
        self.profile_samples = set()
        self.run_id, self.clock_id = uuid.uuid4().hex, clock_id()
        self.method, self.backend = method, backend
        self.algorithm = module_identity() if method == 'gen2' else None
        self.provenance = {key: digest(json.loads(Path(path).read_text()))
                           for key, path in [('calibration_sha256', calibration),
                                             ('pattern_registry_sha256', patterns)]}
        self.submitted_count = self.overwritten_count = self.skipped_cadence_count = 0
        self.expired_input_count = 0
        self.timeout = timeout
        self.max_age = (GEN2_MAX_AGE if method == 'gen2' else 5.) if max_age is None else max_age
        if not math.isfinite(self.max_age) or self.max_age < 0:
            raise ValueError('Invalid measurement age limit')
        if method == 'gen2':
            self.max_age = min(self.max_age, GEN2_MAX_AGE)
        self.condition = threading.Condition()
        self.pending, self.last_sequence = None, None
        self.recording_started = None
        self.metrics = empty_metrics()
        self.finished_recording = None
        self.result = dict(state="starting", detections=[], algorithm=self.algorithm, jpeg_decoder=self.jpeg_decoder)
        self.closed = False
        self.ingress_error = None
        self.dispatcher_state = "starting"
        self.completion_log = None
        self.process = self.connection = None
        self.thread = threading.Thread(target=self.run, name="infrared-latest", daemon=True)
        self.thread.start()

    def submit(self, frame, sequence, source_monotonic=None, recording_started_monotonic=None,
               source_time_kind="application_frame_arrival", runtime_timestamps=None):
        if frame is None or sequence is None:
            return False
        now = time.monotonic()
        metadata = dict(submitted_monotonic=now, source_monotonic=source_monotonic,
                        recording_started_monotonic=recording_started_monotonic, source_time_kind=source_time_kind)
        validate_metadata(metadata)
        if type(sequence) is not int or not 0 <= sequence < 2**64:
            raise ValueError('Invalid source sequence')
        if self.profiling:
            metadata['runtime_timestamps'] = dict(runtime_timestamps or {},
                                                 submit_lock_requested=time.monotonic())
        with self.condition:
            if self.profiling:
                metadata['runtime_timestamps']['submit_lock_acquired'] = time.monotonic()
            if self.closed:
                return False
            self.set_recording_session(recording_started_monotonic)
            if sequence == self.last_sequence:
                return False
            self.last_sequence = sequence
            phase, hz = phase_at(now, recording_started_monotonic)
            metadata.update(phase=phase, configured_admission_hz=hz,
                            admission_unlimited=self.method == 'gen2')
            if self.method == 'gen2':
                if (source_monotonic is None or not 0 <= now-source_monotonic <= self.max_age
                        or recording_started_monotonic is not None and source_monotonic < recording_started_monotonic):
                    self.expired_input_count += 1
                    return False
            self.submitted_count += 1
            self.overwritten_count += int(self.pending is not None)
            # One pending reference: new frames overwrite pending work, never queue.
            self.pending = (frame, sequence, metadata)
            if self.profiling:
                metadata['runtime_timestamps'].update(
                    dispatcher_state_at_notify=self.dispatcher_state,
                    notify_started=time.monotonic())
            self.condition.notify()
            if self.profiling:
                metadata['runtime_timestamps']['notify_completed'] = time.monotonic()
                metadata['runtime_timestamps']['admission_completed'] = time.monotonic()
            return True

    def set_recording_session(self, started):
        validate_metadata(dict(submitted_monotonic=time.monotonic(),
                               recording_started_monotonic=started))
        with self.condition:
            if self.recording_started != started:
                if self.method == 'gen2' and self.recording_started is not None:
                    self.finished_recording = (self.recording_started, self.snapshot()['recording_metrics'])
                self.recording_started = started
                self.pending = None
                self.metrics = empty_metrics()
                self.condition.notify_all()

    def fail_ingress(self, error):
        with self.condition:
            self.ingress_error = str(error)
            self.condition.notify_all()

    def check_errors(self):
        if self.ingress_error is not None:
            raise RuntimeError('Infrared admission failed: '+self.ingress_error)
        if self.completion_log is not None:
            self.completion_log.check()

    def metrics_for_recording(self, started):
        with self.condition:
            if self.recording_started == started:
                return self.snapshot().get('recording_metrics')
            finished = getattr(self, 'finished_recording', None)
            return finished[1] if finished is not None and finished[0] == started else None

    def evidence(self):
        with self.condition:
            provenance = self.result.get('provenance') or self.provenance
        return dict(method=self.method, requested_backend=self.backend, jpeg_decoder=self.jpeg_decoder, algorithm=self.algorithm,
                    provenance=provenance, run_id=self.run_id, clock_id=self.clock_id,
                    source_clock='MONOTONIC', source_time_kind='application_frame_arrival',
                    logging_mode='disabled' if self.log_output is None else 'async_bounded',
                    log_queue_capacity=0 if self.log_output is None else _CompletionLog.capacity,
                    max_age_seconds=self.max_age, log_output=None if self.log_output is None else str(self.log_output))

    def snapshot(self, detail=None):
        def mark(name):
            if detail is not None:
                detail[name] = time.monotonic_ns()
        mark('snapshot_started')
        now = time.monotonic()
        mark('snapshot_lock_requested')
        with self.condition:
            mark('snapshot_lock_acquired')
            result = dict(self.result)
            recording_started = self.recording_started
            metrics = {name: dict(values, by_pattern={key: dict(counts)
                       for key, counts in values['by_pattern'].items()})
                       for name, values in self.metrics.items()}
            mark('snapshot_metrics_copied')
        mark('snapshot_lock_released')
        if (self.method == 'gen2' and result.get('state') not in ('error', 'stopped') and
                result.get('recording_started_monotonic') != recording_started):
            result.update(state='waiting_for_session', detections=[], observations=[])
        result = fresh_result(result, now, self.max_age, self.method == 'gen2')
        mark('snapshot_fresh_completed')
        result.pop('raw_observations', None)  # Audit measurements belong to completion JSONL.
        result.update(run_id=self.run_id, submitted_count=self.submitted_count,
                      overwritten_count=self.overwritten_count,
                      skipped_cadence_count=self.skipped_cadence_count,
                      expired_input_count=self.expired_input_count)
        if self.method == 'gen2':
            phase, hz = phase_at(now, recording_started)
            elapsed = 0. if recording_started is None else max(0., now-recording_started)
            for name, duration in [('startup', min(elapsed, STARTUP_SECONDS)),
                                   ('transition', max(0., elapsed-STARTUP_SECONDS))]:
                start = 0. if recording_started is None else recording_started+(0. if name == 'startup' else STARTUP_SECONDS)
                end = start+duration
                for kind in ('pose', 'position', 'estimated_pose', 'orientation_supported'):
                    last = metrics[name]['last_'+kind+'_monotonic']
                    metrics[name]['max_'+kind+'_gap_seconds'] = max(metrics[name]['max_'+kind+'_gap_seconds'],
                        duration if last is None else max(0., end-last))
                    first = metrics[name]['first_'+kind+'_monotonic']
                    metrics[name]['first_'+kind+'_seconds'] = None if first is None else first-start
                    metrics[name]['effective_'+kind+'_hz'] = metrics[name][kind+'_frames']/duration if duration else None
                for counts in metrics[name]['by_pattern'].values():
                    counts['effective_pose_hz'] = counts['pose_frames']/duration if duration else None
                    counts['effective_position_hz'] = counts['position_frames']/duration if duration else None
                    counts['effective_estimated_pose_hz'] = counts['estimated_pose_frames']/duration if duration else None
                    counts['effective_orientation_supported_hz'] = counts['orientation_supported_frames']/duration if duration else None
                metrics[name].update(duration_seconds=duration,
                    effective_pose_hz=metrics[name]['pose_frames']/duration if duration else None,
                    effective_identity_hz=metrics[name]['identity_frames']/duration if duration else None)
            result.update(recording_phase=phase, configured_admission_hz=hz, admission_unlimited=True,
                          recording_metrics=metrics, algorithm=result.get('algorithm') or self.algorithm)
        mark('snapshot_completed')
        return result

    def receive(self):
        detail = {} if self.profiling else None
        if detail is not None:
            detail['parent_receive_wait_started'] = time.monotonic_ns()
        deadline = time.monotonic()+self.timeout
        while True:
            self.check_errors()
            if self.closed:
                raise EOFError('Infrared worker closed')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Infrared inference exceeded %.1f seconds" % self.timeout)
            if self.connection.poll(min(.1, remaining)):
                break
        if detail is not None:
            detail['parent_receive_ready'] = time.monotonic_ns()
        packet = self.connection.recv_bytes(MAX_PACKET)
        if detail is not None:
            detail['parent_bytes_received'] = time.monotonic_ns()
        header = None
        if packet.startswith(RESULT_TIMING_MAGIC):
            if detail is None or len(packet) <= RESULT_TIMING_HEADER.size:
                raise ValueError('Unexpected/truncated diagnostic result header')
            header = RESULT_TIMING_HEADER.unpack_from(packet)
            packet = packet[RESULT_TIMING_HEADER.size:]
            if not 0 < header[2] <= header[3] <= header[4] <= detail['parent_bytes_received']:
                raise ValueError('Invalid diagnostic result timing order')
        result = json.loads(packet)
        if detail is not None:
            detail['parent_json_parsed'] = time.monotonic_ns()
            detail['result_json_bytes'] = len(packet)
            if header is not None:
                if result.get('sequence') != header[1]:
                    raise ValueError('Diagnostic result sequence mismatch')
                detail.update(worker_serialize_started=header[2], worker_serialize_completed=header[3],
                              worker_send_started=header[4])
            detail.update(result.pop('_runtime_detail_ns', {}))
        self.receive_detail = detail
        return result

    def publish(self, result, timings=None):
        detail = None if timings is None else timings.get('detail_ns')
        def mark(name):
            if detail is not None:
                detail[name] = time.monotonic_ns()
        mark('publish_started')
        mark('publish_lock_requested')
        with self.condition:
            mark('publish_lock_acquired')
            self.result = result
        mark('publish_lock_released')
        if self.output is None and timings is None:
            return
        started = time.monotonic()
        snapshot = self.snapshot(detail) if detail is not None else self.snapshot()
        snapshotted = time.monotonic()
        mark('publish_json_started')
        payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
        mark('publish_json_completed')
        mark('publish_utf8_started')
        payload = payload.encode('utf-8')
        mark('publish_utf8_completed')
        serialized = time.monotonic()
        mark('publish_file_started')
        if self.output is not None:
            self.temporary_output.write_bytes(payload)
        mark('publish_file_written')
        if self.output is not None:
            self.temporary_output.replace(self.output)
        mark('publish_file_replaced')
        if timings is not None:
            source = snapshot.get('source_monotonic')
            age = None if source is None else time.monotonic()-source
            timings.update(published_source_sequence=snapshot.get('source_sequence'),
                published_source_age_ms=None if age is None else age*1000,
                published_fresh=snapshot.get('fresh', False),
                published_detections=len(snapshot.get('detections', [])),
                published_pose=any(d.get('quality', {}).get('pose_usable', False)
                                   for d in snapshot.get('detections', [])),
                published_position=any(d.get('quality', {}).get('position_usable', False)
                                       for d in snapshot.get('detections', [])),
                published_orientation_supported=any(d.get('quality', {}).get('orientation_supported', False)
                                                    for d in snapshot.get('detections', [])),
                published_within_age=age is not None and 0 <= age <= self.max_age)
            timings.update(snapshot_ms=(snapshotted-started)*1000,
                publish_serialization_ms=(serialized-snapshotted)*1000,
                publish_file_ms=(time.monotonic()-serialized)*1000)
        mark('publish_completed')

    def record(self, result):
        if self.log_output is None:
            return
        self.log_output.parent.mkdir(parents=True, exist_ok=True)
        with self.log_output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(result, run_id=self.run_id), ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()

    def record_profile(self, result, frame, timings):
        """Bounded exact inputs: first full/ROI confirmed P03 after 10s per session."""
        session = result.get('recording_started_monotonic')
        mode = result.get('search_mode')
        key = (session, mode)
        if (session is not None and result['source_monotonic']-session >= 10 and
                key not in self.profile_samples and result.get('fresh') and
                any('P03' in d['pattern_id'] for d in result.get('detections', []))):
            started = time.monotonic()
            directory = self.log_output.parent/'infrared-samples'/str(round(session*1e9))
            directory.mkdir(parents=True, exist_ok=True)
            (directory/(str(result['source_sequence'])+'.jpg')).write_bytes(bytes(frame))
            (directory/(str(result['source_sequence'])+'.json')).write_text(
                json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            self.profile_samples.add(key)
            timings['sample_write_ms'] = (time.monotonic()-started)*1000
        path = self.log_output.with_suffix('.profile.jsonl')
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(schema='dohc.infrared.runtime-profile.v1',
                run_id=self.run_id, source_sequence=result['source_sequence'],
                recording_started_monotonic=session, **timings), allow_nan=False)+'\n')

    def record_completion(self, result, frame, timings):
        started = time.monotonic()
        timings.update(logger_dequeued_monotonic=started,
                       logging_queue_wait_ms=(started-timings['log_enqueued_monotonic'])*1000)
        result['logging'] = dict(backpressure_ms=timings['logging_backpressure_ms'],
                                 queue_wait_ms=timings['logging_queue_wait_ms'],
                                 enqueued_monotonic=timings['log_enqueued_monotonic'],
                                 write_started_monotonic=started)
        self.record(result)
        recorded = time.monotonic()
        timings.update(completion_log_ms=(recorded-started)*1000,
                       published_to_recorded_ms=(recorded-timings['published_monotonic'])*1000,
                       recorded_monotonic=recorded)
        if self.profiling:
            self.record_profile(result, frame, timings)

    def take_pending(self, wait):
        with self.condition:
            waited = wait and self.pending is None and not self.closed
            self.dispatcher_state = 'waiting' if waited else 'working'
            if waited:
                self.condition.wait_for(lambda: self.closed or self.pending is not None, timeout=.1)
            resumed = time.monotonic() if self.profiling else None
            if self.closed:
                return None
            work, self.pending = self.pending, None
            taken = time.monotonic() if self.profiling else None
            self.dispatcher_state = 'working'
            if work is not None and self.profiling:
                work[2]['runtime_timestamps'].update(
                    dispatcher_resumed=resumed, dispatcher_task_taken=taken,
                    dispatcher_waited=waited, pending_dequeued=taken)
        return work

    def dispatch(self, work):
        frame, sequence, metadata = work
        source = metadata['source_monotonic']
        with self.condition:
            if self.closed:
                return None
            if self.method == 'gen2':
                if metadata['recording_started_monotonic'] != self.recording_started:
                    return None
                if not 0 <= time.monotonic()-source <= self.max_age:
                    self.expired_input_count += 1
                    return None
            self.dispatcher_state = 'dispatching'
        encoding_started = time.monotonic()
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            import cv2
            if self.jpeg_decoder == 'mpp_rga':
                from .jpeg_decoder import encode_jpeg100
                frame = encode_jpeg100(frame)
            else:
                ok, encoded = cv2.imencode(".png", frame)
                if not ok:
                    raise ValueError("Cannot encode raw infrared input")
                frame = encoded.tobytes()
        metadata['encoding_ms'] = (time.monotonic()-encoding_started)*1000
        ipc_started = time.monotonic()
        if self.profiling:
            trace = metadata['runtime_timestamps']
            trace.update(encoding_started=encoding_started, packet_pack_started=ipc_started)
        packet = encode_packet(sequence, metadata, frame)
        if self.profiling:
            trace['packet_ready'] = time.monotonic()
        with self.condition:
            if self.closed or (self.method == 'gen2' and
                    metadata['recording_started_monotonic'] != self.recording_started):
                return None
            if self.method == 'gen2' and not 0 <= time.monotonic()-source <= self.max_age:
                self.expired_input_count += 1
                return None
        self.connection.send_bytes(packet)
        if self.profiling:
            trace['packet_sent'] = time.monotonic()
        return (frame, sequence, metadata), ipc_started

    def run(self):
        local, child = socket.socketpair()
        self.connection = Connection(local.detach())
        work = inflight = dispatching = None
        unlogged_completion = None
        try:
            if self.output is not None:
                self.output.parent.mkdir(parents=True, exist_ok=True)
            if self.log_output is not None:
                self.completion_log = _CompletionLog(self.record_completion)
            env = os.environ.copy()
            for name in ("PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
                env.pop(name, None)
            if __package__ == 'infrared':
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            if self.profile_override is not None:
                env['DOHC_IR_TRACKER_PROFILE'] = 'true' if self.profiling else 'false'
            self.process = subprocess.Popen(self.arguments+["--fd", str(child.fileno())],
                pass_fds=(child.fileno(),), env=env, stdin=subprocess.DEVNULL)
            child.close()
            ready = self.receive()
            if ready.get("state") != "ready":
                raise RuntimeError(ready.get("error", "Infrared initialization failed"))
            self.publish(ready)
            while True:
                self.check_errors()
                if inflight is None:
                    work = self.take_pending(wait=True)
                    if self.closed:
                        break
                    if work is None:
                        self.publish(dict(self.result))
                        continue
                    inflight = self.dispatch(work)
                    if inflight is None:
                        work = None
                        self.publish(dict(self.result))
                        continue
                work, ipc_started = inflight
                inflight = None
                frame, sequence, metadata = work
                trace = metadata.get('runtime_timestamps')
                self.dispatcher_state = 'receiving'
                result = self.receive()
                received = time.monotonic()
                detail = self.receive_detail if self.profiling else None
                def mark(name):
                    if detail is not None:
                        detail[name] = time.monotonic_ns()
                mark('parent_result_received')
                self.dispatcher_state = 'validating'
                decode_profile = result.pop('decode_profile_ns', None)
                if self.profiling:
                    result.setdefault('runtime_timestamps', {}).update(trace)
                if result.get("state") == "error":
                    raise RuntimeError(result.get("error", "Infrared worker failed"))
                if result.get("sequence") != sequence:
                    raise ValueError("Infrared result sequence mismatch")
                result.update(run_id=self.run_id, source_sequence=sequence, method=self.method,
                              clock_id=self.clock_id, requested_backend=self.backend,
                              submitted_count=self.submitted_count,
                              overwritten_count=self.overwritten_count,
                              skipped_cadence_count=self.skipped_cadence_count,
                              expired_input_count=self.expired_input_count)
                if self.method == 'gen2':
                    result.setdefault('algorithm', self.algorithm)
                    # Retain complete pixel measurements in the audit log even if expired.
                    result.setdefault('raw_observations', result.get('observations', []))
                mark('parent_fields_completed')
                result = fresh_result(result, time.monotonic(), self.max_age, self.method == 'gen2')
                mark('parent_fresh_completed')
                mark('parent_session_lock_requested')
                with self.condition:
                    mark('parent_session_lock_acquired')
                    if self.method == 'gen2' and metadata['recording_started_monotonic'] != self.recording_started:
                        result.update(state='superseded_session', detections=[], observations=[], fresh=False)
                    mark('parent_session_checked')
                    if self.method == 'gen2' and self.recording_started is not None and result['state'] == 'ready':
                        phase, _ = phase_at(result['completed_monotonic'], self.recording_started)
                        metric = self.metrics[phase]
                        count_completion(metric, result, self.recording_started+
                                         (STARTUP_SECONDS if phase == 'transition' else 0.))
                    mark('parent_metrics_completed')
                mark('parent_session_lock_released')
                diagnostic = result.pop('_profile', None)
                timings = dict(ipc_roundtrip_ms=(received-ipc_started)*1000,
                               received_monotonic=received)
                if detail is not None:
                    timings.update(timing_version='dev17-overlap-v1', detail_ns=detail,
                                   decode_profile_ns=decode_profile)
                if self.completion_log is not None:
                    unlogged_completion = sequence
                # The child has released n; send at most one n+1 before publishing n.
                mark('next_dispatch_started')
                if detail is not None:
                    detail['next_dispatched_sequence'] = None
                if self.method == 'gen2' and not self.closed:
                    dispatching = self.take_pending(wait=False)
                    if dispatching is not None:
                        inflight = self.dispatch(dispatching)
                    if detail is not None:
                        detail['next_dispatched_sequence'] = None if inflight is None else inflight[0][1]
                    dispatching = None
                mark('next_dispatch_completed')
                self.dispatcher_state = 'publishing'
                started = time.monotonic()
                self.publish(result, timings if self.profiling else None)
                published = time.monotonic()
                timings.update(publish_ms=(published-started)*1000, published_monotonic=published)
                self.dispatcher_state = 'logging'
                if self.completion_log is not None:
                    logged = dict(result)
                    if self.profiling:
                        logged['profile'] = diagnostic
                    self.completion_log.put(logged, frame if self.profiling else None, timings)
                    unlogged_completion = None
                work = None
            if self.completion_log is not None:
                self.completion_log.close()
        except Exception as exc:
            failure = dict(state="stopped" if self.closed else "error", error=str(exc),
                           clock_id=self.clock_id, detections=[], method=self.method, jpeg_decoder=self.jpeg_decoder, fresh=False,
                           algorithm=self.algorithm, completed_monotonic=time.monotonic(), run_id=self.run_id)
            if work is not None:
                failure.update(sequence=work[1], source_sequence=work[1], **work[2])
            if inflight is not None:
                failure['inflight_source_sequence'] = inflight[0][1]
            if dispatching is not None:
                failure['dispatching_source_sequence'] = dispatching[1]
            if self.completion_log is not None:
                try:
                    self.completion_log.close()
                except Exception as log_error:
                    failure.update(log_error=str(log_error),
                                   unwritten_log_sequences=self.completion_log.unwritten_sequences())
                    failure['state'] = 'error'
            if unlogged_completion is not None:
                failure.setdefault('unwritten_log_sequences', []).append(unlogged_completion)
            try:
                if self.completion_log is None or not self.completion_log.thread.is_alive():
                    self.record(failure)
            except Exception as log_error:
                failure['log_error'] = str(log_error)
            self.publish(failure)
        finally:
            child.close()
            self.connection.close()
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

    def close(self):
        with self.condition:
            self.closed, self.pending = True, None
            self.condition.notify()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise TimeoutError('Infrared dispatcher did not stop')
        if self.result.get('state') != 'error':
            self.publish(dict(state="stopped", detections=[]))


def main():
    import argparse
    import cv2
    from .jpeg_decoder import JPEGDecoder
    from .methods import METHODS, create_detector
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "gpu-npu"), required=True)
    parser.add_argument("--npu-model", type=Path)
    parser.add_argument("--method", choices=METHODS, default="gen2")
    parser.add_argument('--jpeg-decoder', choices=('cpu', 'mpp_rga'))
    args = parser.parse_args()
    decoder_backend = args.jpeg_decoder or os.environ.get('DOHC_IR_TRACKER_JPEG_DECODER',
        'mpp_rga' if args.method == 'gen2' else 'cpu')
    if args.method != 'gen2' and decoder_backend != 'cpu':
        parser.error('MPP/RGA image sources require gen2')
    cv2.setNumThreads(1)
    connection = Connection(args.fd)
    detector = decoder = None
    try:
        # Board libraries require Mali initialization before MPP/RGA is loaded.
        detector = create_detector(args.method, json.loads(args.calibration.read_text()),
                                   json.loads(args.patterns.read_text()), args.backend, args.npu_model)
        decoder = JPEGDecoder(decoder_backend)
        if args.method == 'gen2':
            detector.require_online_patterns()
        connection.send_bytes(json.dumps(dict(state="ready", detections=[], method=args.method,
            algorithm=getattr(detector, 'algorithm', None), provenance=getattr(detector, 'provenance', None),
            backend=detector.extractor.name, jpeg_decoder=decoder_backend)).encode())
        previous_session, previous_source = object(), None
        while True:
            packet = connection.recv_bytes(MAX_PACKET)
            received = time.monotonic() if profiling_enabled() else None
            sequence, metadata, encoded = decode_packet(packet)
            started = time.monotonic()
            trace = metadata.get('runtime_timestamps')
            if trace is not None:
                trace.update(worker_packet_received=received, worker_packet_parsed=started)
            source, session = metadata['source_monotonic'], metadata['recording_started_monotonic']
            if args.method == 'gen2' and (session != previous_session or
                    source is not None and previous_source is not None and source < previous_source):
                detector.reset()
            previous_session, previous_source = session, source
            if args.method == 'gen2' and (source is None or not 0 <= started-source <= GEN2_MAX_AGE
                    or session is not None and source < session):
                result = dict(detections=[], observations=[], processing_ms=0.,
                              skip_reason='expired_input', method=args.method)
            else:
                if trace is not None:
                    trace['worker_decode_started'] = time.monotonic()
                with decoder.decode(encoded) as frame:
                    if trace is not None:
                        trace['worker_decode_completed'] = time.monotonic()
                    if args.method == 'gen2':
                        result = detector.detect(frame, timestamp=source,
                                                 recording_started_monotonic=session)
                    else:
                        result = detector.detect(frame.image, timestamp=source if source is not None else metadata['submitted_monotonic'])
                    if trace is not None:
                        trace['worker_detect_completed'] = time.monotonic()
                    result.update(frame.metrics)
                if trace is not None:
                    trace['worker_frame_released'] = time.monotonic()
            result.update(metadata, state="ready", jpeg_decoder=decoder_backend, sequence=sequence, started_monotonic=started,
                          completed_monotonic=time.monotonic())
            detail = {} if trace is not None else None
            if detail is not None:
                detail['worker_result_prepare_started'] = time.monotonic_ns()
            if args.method == 'gen2':
                detector.retain_startup_seed(result)
                result['raw_observations'] = result.get('observations', [])
                result = fresh_result(result, time.monotonic(), GEN2_MAX_AGE, True)
            if detail is not None:
                detail['worker_result_prepare_completed'] = time.monotonic_ns()
                result['_runtime_detail_ns'] = detail
            send_result(connection, result, diagnostic=detail is not None)
    except EOFError:
        pass
    except Exception as exc:
        connection.send_bytes(json.dumps(dict(state="error", error=str(exc), detections=[])).encode())
    finally:
        try:
            if decoder is not None:
                decoder.close()
        finally:
            if detector is not None:
                detector.close()
            connection.close()


if __name__ == "__main__":
    main()
