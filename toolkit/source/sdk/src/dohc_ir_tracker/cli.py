"""Standalone CLI. Live inference is provided by the SDK, not a camera service."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

from .tracker import Tracker, load_config, require_platform


def doctor(config=None):
    checks = dict(platform=platform.platform(), python=platform.python_version(),
                  hardware_execution='not_attempted', device_cid='未读取', device_ip='未读取',
                  device_version='未读取')
    try:
        require_platform()
        if config is not None:
            cfg = load_config(config)
        else:
            cfg = None
        from ._core.version import module_identity
        from ._core.assignment import verify_native_build, LIBRARY
        checks['algorithm'] = module_identity()
        verify_native_build(LIBRARY)
        from ._core.opencl_extract import OpenCLExtractor
        from ._core.jpeg_decoder import JPEGDecoder
        # Vendor stack requires Mali before loading MPP/RGA.
        extractor = OpenCLExtractor()
        try:
            checks['gpu'] = extractor.device
            if 'mali' not in extractor.device.lower():
                raise RuntimeError('This package requires the DOHC Mali GPU')
            decoder = JPEGDecoder('mpp_rga')
            try:
                checks['codec'] = decoder.identity
            finally:
                decoder.close()
        finally:
            extractor.close()
        if cfg:
            from ._core.gen2 import Gen2Tracker
            detector = Gen2Tracker(json.loads(cfg['calibration'].read_text()), json.loads(cfg['patterns'].read_text()))
            detector.require_online_patterns()
        checks.update(ok=True, hardware_execution='initialization_only; no image inference')
    except Exception as exc:
        checks.update(ok=False, error=str(exc))
    return checks


def inspect_image(config, image):
    require_platform()
    cfg = load_config(config)
    from ._core.methods import create_detector
    from ._core.jpeg_decoder import JPEGDecoder
    detector = create_detector('gen2', json.loads(cfg['calibration'].read_text()),
                               json.loads(cfg['patterns'].read_text()), 'gpu')
    decoder = None
    try:
        decoder = JPEGDecoder('mpp_rga')
        with decoder.decode(Path(image).read_bytes()) as frame:
            result = detector.search(frame)
            result.update(frame.metrics, execution_mode='single_image_diagnostic',
                          identity_confirmation_performed=False, live_freshness_evaluated=False)
        return result
    finally:
        if decoder:
            decoder.close()
        detector.close()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    entries = manifest['entries']
    previous, previous_sequence, sequences = -1., -1, set()
    for entry in entries:
        stamp, sequence = entry['relative_seconds'], entry['source_sequence']
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or
                not math.isfinite(stamp) or stamp < 0 or stamp <= previous or
                type(sequence) is not int or not previous_sequence < sequence < 2**64 or sequence in sequences):
            raise ValueError('Manifest requires increasing finite offsets and unique source sequences')
        previous = stamp
        sequences.add(sequence)
        previous_sequence = sequence
        image = (path.parent/entry['file']).resolve(strict=True)
        image.relative_to(path.parent)  # A manifest may not escape its input directory.
        if hash_file(image) != entry['sha256']:
            raise ValueError('JPEG hash mismatch: '+entry['file'])
    if not entries:
        raise ValueError('Empty replay manifest')
    return manifest


def prepare_session(session, output, calibration):
    """Reconstruct complete DOHC MP4 sessions, preserving BOOTTIME differences."""
    import cv2
    from ._core.jpeg_decoder import encode_jpeg100
    session, output = Path(session).resolve(strict=True), Path(output)
    times_path = session/'times/cam0.json'
    timing = json.loads(times_path.read_text())
    if timing.get('unit') != 'ns' or timing.get('clock') != 'CLOCK_BOOTTIME':
        raise ValueError('DOHC replay expects CLOCK_BOOTTIME nanosecond metadata')
    states = [json.loads(line) for line in (session/'states.jsonl').read_text().splitlines()]
    if not states:
        raise ValueError('Missing capture states')
    origin = states[0]['capture_boottime_ns']
    sequences = [s['capture_sequences']['cam0'] for s in states
                 if s.get('capture_sequences', {}).get('cam0') is not None]
    stamps = timing['timestamps_ns']
    if len(stamps) != len(sequences) or len(set(sequences)) != len(sequences):
        raise ValueError('Video metadata and source sequences differ')
    output.mkdir(parents=True, exist_ok=False)
    entries, index, sources = [], 0, {}
    for video in sorted((session/'cam0').glob('*.mp4')):
        sources[str(video.relative_to(session))] = hash_file(video)
        capture = cv2.VideoCapture(str(video))
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape != (2160,3840,3) or index >= len(stamps):
                    raise ValueError('Unexpected image size or excess video frames')
                name = '%08d.jpg' % index
                data = encode_jpeg100(image)
                (output/name).write_bytes(data)
                entries.append(dict(file=name, frame_index=index, source_sequence=sequences[index],
                    relative_seconds=(stamps[index]-origin)/1e9,
                    sha256=hashlib.sha256(data).hexdigest()))
                index += 1
        finally:
            capture.release()
    if index != len(stamps):
        raise ValueError('Full video decode count differs from timestamp metadata')
    sources['times/cam0.json'] = hash_file(times_path)
    sources['states.jsonl'] = hash_file(session/'states.jsonl')
    manifest = dict(schema='dohc.ir.common-input.v1', source_path=str(session), source_files=sources,
        input_kind='MP4_reconstructed_Q100_JPEG', source_clock='CLOCK_BOOTTIME_relative_only',
        warning='Not original camera JPEG; preparation excluded from inference timing',
        calibration=calibration, entries=entries)
    path = output/'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    validate_manifest(path)
    return path


def replay(config, source, output):
    require_platform()
    cfg = load_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = Path(source).resolve(strict=True)
    # NAS preparation must be explicit and verified before this command is invoked.
    if str(source).startswith('/mnt/Datasets/'):
        raise ValueError('Copy and verify NAS inputs to local storage before replay')
    manifest_path = (prepare_session(source, output/'input', json.loads(cfg['calibration'].read_text()))
                     if source.is_dir() else source)
    manifest = validate_manifest(manifest_path)
    preparation_seconds = time.perf_counter()-started
    resolved = {k:str(v) if isinstance(v, Path) else v for k,v in cfg.items()}
    resolved['output_dir'] = str(output/'sessions')
    config_path = output/'resolved-config.json'
    config_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2)+'\n')
    tracker = Tracker(config_path)
    submitted = []
    try:
        tracker.start()
        origin = tracker._start
        for entry in manifest['entries']:
            data = (manifest_path.parent/entry['file']).read_bytes()
            due = origin+entry['relative_seconds']
            while time.monotonic() < due:
                time.sleep(min(.02, max(0.,due-time.monotonic())))
            admitted = tracker.submit(data, entry['source_sequence'], due)
            submitted.append(dict(source_sequence=entry['source_sequence'],
                                  relative_seconds=entry['relative_seconds'], admitted=admitted))
        deadline = time.monotonic()+2.
        while time.monotonic() < deadline:
            row = tracker.latest_result()
            if row.get('source_sequence') == manifest['entries'][-1]['source_sequence']:
                break
            time.sleep(.01)
    finally:
        tracker.close()
    log = tracker.output_directory/'results.jsonl'
    shutil.copy2(log, output/'results.jsonl')
    (output/'admission.json').write_text(json.dumps(submitted, indent=2)+'\n')
    (output/'replay-info.json').write_text(json.dumps(dict(preparation_seconds=preparation_seconds,
        input_manifest=str(manifest_path), input_manifest_sha256=hash_file(manifest_path),
        execution='paced replay with real budgets and latest-frame overwrite; not camera capture FPS'), indent=2)+'\n')
    from .report import generate
    return generate(output/'results.jsonl', output/'report', manifest_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='dev.19.4 DOHC硬件加速识别工具包')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('doctor', help='检查DOHC硬件及依赖')
    check.add_argument('--config', type=Path)
    image = sub.add_parser('image', help='单张JPEG诊断，不伪造两帧确认')
    image.add_argument('--config', type=Path, required=True)
    image.add_argument('--input', type=Path, required=True)
    image.add_argument('--output', type=Path)
    play = sub.add_parser('replay', help='原始JPEG清单或DOHC录像会话原速回放')
    play.add_argument('--config', type=Path, required=True)
    play.add_argument('--input', type=Path, required=True)
    play.add_argument('--output', type=Path, required=True)
    report = sub.add_parser('report', help='结果文件生成CSV、HTML和可选标注视频')
    report.add_argument('--results', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    report.add_argument('--input-manifest', type=Path)
    report.add_argument('--video', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            value = doctor(args.config)
        elif args.command == 'image':
            value = inspect_image(args.config, args.input)
        elif args.command == 'replay':
            value = replay(args.config, args.input, args.output)
        else:
            from .report import generate
            value = generate(args.results, args.output, args.input_manifest, args.video)
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
        if args.command == 'image' and args.output:
            with args.output.open('x') as stream:
                stream.write(text)
        else:
            print(text, end='')
        return 1 if args.command == 'doctor' and not value['ok'] else 0
    except Exception as exc:
        print(json.dumps(dict(state='error', error=str(exc), detections=[]), ensure_ascii=False), file=sys.stderr)
        return 1
