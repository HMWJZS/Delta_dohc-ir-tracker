"""Small lifecycle wrapper over the frozen latest-frame worker."""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import threading
import time


def require_platform():
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('This distribution requires Linux ARM64 / RK3588; CPU fallback is not enabled')
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError('This offline distribution requires the DOHC system CPython 3.8 ABI')


def load_config(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    allowed = {'calibration', 'patterns', 'output_dir', 'diagnostics', 'optical_profile_confirmed'}
    if not isinstance(value, dict) or set(value)-allowed:
        raise ValueError('Unknown configuration keys; algorithm thresholds are frozen')
    if value.get('optical_profile_confirmed') is not True:
        raise ValueError('Confirm the calibration, physical camera, V1 lens and 940nm filter before use')
    for key in ('calibration', 'patterns'):
        value[key] = (path.parent/Path(value[key])).resolve(strict=True)
    calibration = json.loads(value['calibration'].read_text())
    if (calibration.get('camera_model') != 'opencv_fisheye' or
            (calibration.get('image_width'), calibration.get('image_height')) != (3840, 2160)):
        raise ValueError('Native 3840x2160 OpenCV fisheye calibration is required')
    if type(value.get('diagnostics', False)) is not bool:
        raise ValueError('diagnostics must be boolean')
    if value.get('output_dir') is not None:
        value['output_dir'] = (path.parent/Path(value['output_dir'])).resolve()
    value.setdefault('diagnostics', False)
    value.setdefault('output_dir', None)
    if value['diagnostics'] and value.get('output_dir') is None:
        raise ValueError('diagnostics requires output_dir')
    return value


def _create_worker(config, directory):
    from ._core.runtime import InfraredWorker
    return InfraredWorker(sys.executable, config['calibration'], config['patterns'],
        backend='gpu', method='gen2', jpeg_decoder='mpp_rga',
        output=None if directory is None else directory/'latest.json',
        log_output=None if directory is None else directory/'results.jsonl',
        profiling=config['diagnostics'])


class Tracker:
    """One worker and one overwriteable pending reference, never a frame FIFO.

    Live timestamps must use this machine's time.monotonic(), not exposure,
    epoch time or an old recording's clock. submit=True means admitted only.
    """
    def __init__(self, config):
        self.config = load_config(config)
        self._lock = threading.RLock()
        self._worker = None
        self._closed = False
        self._start = None
        self._sequence = -1
        self.output_directory = None
        self.last_metrics = None
        self._terminal = dict(schema='dohc.infrared.v1', state='stopped', fresh=False,
                              detections=[], observations=[])

    def start(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Tracker is closed')
            if self._worker is not None:
                raise RuntimeError('Session already started; stop it before restarting')
            require_platform()
            directory = self.config.get('output_dir')
            if directory is not None:
                import uuid
                directory = directory/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
                directory.mkdir(parents=True, exist_ok=False)
            worker = _create_worker(self.config, directory)
            try:
                deadline = time.monotonic()+worker.timeout
                while True:
                    state = worker.snapshot()
                    if state.get('state') == 'error':
                        raise RuntimeError(state.get('error', 'Hardware initialization failed'))
                    if state.get('state') == 'ready':
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Hardware worker initialization timed out')
                    time.sleep(.01)
                started = time.monotonic()
                worker.set_recording_session(started)
            except BaseException:
                worker.close()
                raise
            if directory is not None:
                try:
                    from ._core.version import module_identity
                    manifest = dict(algorithm=module_identity(), session_started_monotonic=started,
                        source_clock='MONOTONIC',
                        input_files={key:hashlib.sha256(self.config[key].read_bytes()).hexdigest()
                                     for key in ('calibration','patterns')})
                    (directory/'session.json').write_text(json.dumps(manifest, indent=2)+'\n')
                except BaseException:
                    worker.close()
                    raise
            self._worker, self._start = worker, started
            self._sequence = -1
            self.output_directory = directory
            self.last_metrics = None
            self._terminal = dict(schema='dohc.infrared.v1', state='waiting_for_session',
                                  fresh=False, detections=[], observations=[])
            return self

    def submit(self, jpeg_bytes, sequence=None, source_monotonic=None):
        received = time.monotonic()
        if not isinstance(jpeg_bytes, (bytes, bytearray, memoryview)):
            raise TypeError('submit requires original JPEG bytes')
        # Own mutable producer buffers before asynchronous dispatch can read them.
        frame = jpeg_bytes if isinstance(jpeg_bytes, bytes) else bytes(jpeg_bytes)
        if not frame.startswith(b'\xff\xd8') or len(frame) > 32*1024*1024-8192:
            raise ValueError('Invalid JPEG header or input exceeds the IPC budget')
        stamp = received if source_monotonic is None else source_monotonic
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError('Source time must be a finite nonnegative MONOTONIC value')
        with self._lock:
            if self._worker is None:
                raise RuntimeError('Call start() before submit()')
            self._raise_worker_error()
            sequence = self._sequence+1 if sequence is None else sequence
            if type(sequence) is not int or not 0 <= sequence < 2**64:
                raise ValueError('Sequence must be a uint64 integer')
            if sequence <= self._sequence:
                return False
            admitted = self._worker.submit(frame, sequence, source_monotonic=stamp,
                recording_started_monotonic=self._start,
                source_time_kind='sdk_frame_receipt' if source_monotonic is None else 'application_frame_arrival')
            self._sequence = sequence
            return admitted

    def _raise_worker_error(self):
        state = self._worker.snapshot()
        if state.get('state') == 'error':
            raise RuntimeError(state.get('error', 'Infrared worker failed'))

    def latest_result(self):
        with self._lock:
            if self._worker is None:
                return copy.deepcopy(self._terminal)
            self._raise_worker_error()
            return copy.deepcopy(self._worker.snapshot())

    def stop(self):
        with self._lock:
            worker, self._worker = self._worker, None
            if worker is None:
                return
            try:
                worker.close()
                self.last_metrics = worker.metrics_for_recording(self._start)
            finally:
                self._start = None
                self._terminal = dict(schema='dohc.infrared.v1', state='stopped', fresh=False,
                                      detections=[], observations=[])
            if self.output_directory is not None:
                (self.output_directory/'session-summary.json').write_text(json.dumps(
                    self.last_metrics, ensure_ascii=False, allow_nan=False, indent=2)+'\n')

    def close(self):
        with self._lock:
            try:
                self.stop()
            finally:
                self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
