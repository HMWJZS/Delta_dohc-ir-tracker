"""Module identity is independent of system VERSION and the result schema."""
import hashlib
import json
from pathlib import Path
import subprocess

from .config import CONFIG

ALGORITHM = dict(id="dohc-ir-ug", name="DOHC IR Tracker Gen2",
                 generation=2, version="2.0.0-dev.19.4")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def module_identity():
    module = Path(__file__).resolve().parent
    if module.name == '_core':
        metadata = json.loads((module.parent/'build-manifest.json').read_text())
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in module.iterdir() if p.suffix in ('.py', '.csv', '.json')}
        if hashes != metadata['core_sha256']:
            raise ValueError('Standalone core differs from build manifest')
        return dict(ALGORITHM, config_sha256=digest(CONFIG),
                    source_commit=metadata['source_commit'], source_dirty=metadata['source_dirty'],
                    source_sha256=digest(hashes), source_files=hashes,
                    upstream_commit=metadata['upstream_commit'],
                    package_version=metadata['package_version'], system_version=None,
                    system_release_id=None)
    root = module.parents[1]
    release = root/'RELEASE.json'
    metadata = json.loads(release.read_text()) if release.exists() else {}
    commit = metadata.get('commit')
    dirty = metadata.get("source_dirty")
    if commit is None:
        try:
            dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'],
                stderr=subprocess.DEVNULL, timeout=2, text=True).strip())
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL, timeout=2, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
    hashes = {name: hashlib.sha256((module/name).read_bytes()).hexdigest()
              for name in ('version.py', 'gen2.py', 'geometry.py', 'detector.py',
                           'led_coordinates.csv', 'methods.py', 'runtime.py', 'config.py',
                           'pose_resolution.py', 'roi_motion.py', 'roi_association.py', 'profiling.py', 'opencl_extract.py',
                           'image_source.py', 'jpeg_decoder.py', 'assignment.py')}
    for name in ('camera_thread.py', 'camera_startup.py'):
        hashes['dohc/'+name] = hashlib.sha256((root/'dohc'/name).read_bytes()).hexdigest()
    for directory in ('infrared_codec', 'infrared_assignment'):
        for path in sorted((root/'native'/directory).glob('*')):
            if path.suffix in ('.cpp', '.h', '.txt', '.so', '.json'):
                hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(ALGORITHM, config_sha256=digest(CONFIG), source_commit=commit, source_dirty=dirty,
                system_version=(root/'VERSION').read_text().strip(), source_sha256=digest(hashes),
                source_files=hashes, system_release_id=metadata.get('release_id'))
