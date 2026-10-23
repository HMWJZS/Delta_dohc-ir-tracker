"""R5 batch assignment with process-local loading and frame-local workspaces."""
import ctypes as ct
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from .resources import resource_root

LIBRARY = resource_root()/'native/infrared_assignment/libdohc_ir_assignment.so'


def verify_native_build(path, machine=None):
    path = Path(path)
    metadata = json.loads(path.with_name('build-info.json').read_text())
    data = path.read_bytes()
    expected = machine if machine is not None else {
        'x86_64': 62, 'amd64': 62, 'aarch64': 183, 'arm64': 183,
    }.get(platform.machine().lower())
    if (expected is None or len(data) < 20 or data[:6] != b'\x7fELF\x02\x01'
            or int.from_bytes(data[18:20], 'little') != expected):
        raise ValueError('Native assignment ELF architecture mismatch')
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in path.parent.iterdir() if p.suffix in ('.h', '.cpp', '.txt')}
    if (not isinstance(metadata, dict) or metadata.get('schema') != 'dohc.ir.assignment-build.v1'
            or metadata.get('abi') != 1 or not sources
            or metadata.get('library_sha256') != hashlib.sha256(data).hexdigest()
            or metadata.get('source_sha256') != sources):
        raise ValueError('Native assignment build manifest/library/source mismatch')
    return metadata


@lru_cache(maxsize=1)
def load_native():
    try:
        verify_native_build(LIBRARY)
        lib = ct.CDLL(str(LIBRARY))
        lib.ir_assignment_abi.restype = ct.c_uint32
        if lib.ir_assignment_abi() != 1:
            raise ValueError('Native assignment ABI mismatch')
        lib.ir_assign_f64.argtypes = [ct.c_int, ct.c_int]+[ct.c_void_p]*4
        lib.ir_assign_f64.restype = ct.c_int
        return lib, ''
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        return None, str(exc)


class AssignmentWorkspace:
    """Returned views are valid until the next assignment in this frame."""

    def __init__(self):
        self.lib, self.reason = load_native()
        self.buffers = {}
        self.native_calls = self.numpy_calls = 0
        self.backend = 'native' if self.lib is not None else 'numpy'

    def assign(self, predictions, observed):
        batches, points = len(predictions), len(observed)
        if (observed.shape != (points, 2) or batches < 1 or points < 1
                or any(p.shape != (points, 2) for p in predictions)):
            raise ValueError('Assignment requires BxNx2 predictions and Nx2 observations')
        native = (self.lib is not None and batches <= 8 and points <= 12
                  and observed.dtype == np.float64
                  and all(p.dtype == np.float64 for p in predictions))
        if native:
            if points not in self.buffers:
                predicted = np.empty((8, points, 2), np.float64)
                indices = np.empty((8, points), np.int64)
                distances = np.empty((8, points), np.float64)
                self.buffers[points] = (predicted, indices, distances,
                    predicted.ctypes.data, indices.ctypes.data, distances.ctypes.data)
            predicted, indices, distances, pptr, iptr, dptr = self.buffers[points]
            for index, value in enumerate(predictions):
                predicted[index] = value
            observed = np.require(observed, requirements=['C', 'A'])
            code = self.lib.ir_assign_f64(batches, points, pptr, observed.ctypes.data, iptr, dptr)
            if code == 0:
                self.native_calls += 1
                self.backend = 'mixed' if self.numpy_calls else 'native'
                return indices[:batches], distances[:batches]
            self.lib, self.reason = None, 'Native assignment rejected input: '+str(code)
        elif self.lib is not None:
            self.reason = 'Unsupported native assignment shape/dtype'
        self.numpy_calls += 1
        self.backend = 'mixed' if self.native_calls else 'numpy'
        distance = np.linalg.norm(np.asarray(predictions)[:, :, None]-observed[None, None], axis=3)
        indices = distance.argmin(axis=2)
        return indices, np.take_along_axis(distance, indices[:, :, None], axis=2)[:, :, 0]
