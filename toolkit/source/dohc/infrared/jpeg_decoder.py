"""CPU reference or persistent MPP/RGA decoder, owned by the inference process."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .image_source import crop_bgr
from .resources import resource_root
from .profiling import span, enabled as profiling_enabled

JPEG_QUALITY = 100
DECODERS = ('cpu', 'mpp_rga')
DECODE_BOUNDARIES = ('native_started', 'jpeg_checked', 'buffers_ready', 'dma_write_started',
    'packet_copied', 'dma_write_completed', 'task_objects_ready', 'input_ready',
    'task_submitted', 'output_ready', 'output_extracted', 'output_checked', 'native_completed')


class DecodeProfile(ct.Structure):
    _fields_ = [('boundary_ns', ct.c_uint64*len(DECODE_BOUNDARIES))]


class FrameInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in
                ('width', 'height', 'hstride', 'vstride', 'format', 'colorspace', 'range')]


class CropStats(ct.Structure):
    _fields_ = [('rga_ms', ct.c_double), ('map_copy_ms', ct.c_double),
                ('converted_pixels', ct.c_uint64)]


def encode_jpeg100(image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError('Cannot encode infrared JPEG')
    return encoded.tobytes()


def verify_native_build(path):
    metadata = json.loads(path.with_name('build-info.json').read_text())
    if (metadata.get('schema') != 'dohc.ir.codec-build.v1' or metadata.get('abi') != 1 or
            metadata.get('library_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError('Native codec build manifest/library mismatch')
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in path.parent.iterdir() if p.suffix in ('.h', '.cpp', '.txt')}
    if not sources or metadata.get('source_sha256') != sources:
        raise ValueError('Native codec sources changed; rebuild MPP/RGA library')
    return metadata


class BGRFrame:
    def __init__(self, image, elapsed_ms):
        self.image = image
        self.shape, self.dtype, self.ndim, self.size = image.shape, image.dtype, image.ndim, image.size
        self.metrics = dict(jpeg_decoder='cpu', decode_ms=elapsed_ms,
                            decode_scope='cpu_imdecode_to_full_bgr')

    def crop_bgr(self, rectangle):
        if self.image is None:
            raise RuntimeError('Frame is closed')
        return crop_bgr(self.image, rectangle)

    def close(self):
        self.image = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HardwareFrame(BGRFrame):
    def __init__(self, decoder, info, elapsed_ms):
        self.decoder, self.info = decoder, info
        self.shape = (int(info.height), int(info.width), 3)
        self.dtype, self.ndim, self.size = np.dtype('uint8'), 3, int(info.width)*int(info.height)*3
        self.patches = []
        self.metrics = dict(jpeg_decoder='mpp_rga', decode_ms=elapsed_ms,
            decode_scope='mpp_jpeg_to_yuv_including_packet_copy_and_wait', mpp_decode_ms=elapsed_ms,
            rga_ms=0., map_copy_ms=0., converted_pixels=0, bgr_patch_bytes=0, crop_calls=0,
            native_frame={name: int(getattr(info, name)) for name, _ in info._fields_},
            native_library=decoder.identity)

    def crop_bgr(self, rectangle):
        if self.decoder is None:
            raise RuntimeError('Frame is closed')
        x0, y0, x1, y1 = map(int, rectangle)
        h, w = self.shape[:2]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('Invalid hardware BGR crop')
        for (a, b, c, d), patch in self.patches:
            if a <= x0 < x1 <= c and b <= y0 < y1 <= d:
                return patch[y0-b:y1-b, x0-a:x1-a]
        patch = np.empty((y1-y0, x1-x0, 3), dtype=np.uint8)
        stats = CropStats()
        with span('rga_crop_convert'):
            self.decoder.check(self.decoder.lib.ir_decoder_crop_bgr(self.decoder.handle,
                x0, y0, x1-x0, y1-y0, patch.ctypes.data_as(ct.c_void_p),
                patch.strides[0], ct.byref(stats)))
        self.patches.append((tuple(rectangle), patch))
        for name in ('rga_ms', 'map_copy_ms', 'converted_pixels'):
            self.metrics[name] += getattr(stats, name)
        self.metrics['bgr_patch_bytes'] += patch.nbytes
        self.metrics['crop_calls'] += 1
        return patch

    def close(self):
        if self.decoder is not None:
            self.decoder.lib.ir_decoder_release_frame(self.decoder.handle)
            self.decoder.active = None
            self.decoder = None
            self.patches.clear()


class JPEGDecoder:
    def __init__(self, backend='mpp_rga', library=None):
        if backend not in DECODERS:
            raise ValueError('Unknown JPEG decoder: '+str(backend))
        self.backend, self.active, self.handle = backend, None, ct.c_void_p()
        self.profiling = profiling_enabled()
        self.closed = False
        if backend == 'cpu':
            return
        path = Path(library) if library is not None else (
            resource_root()/'native/infrared_codec/libdohc_ir_codec.so')
        build_manifest = verify_native_build(path) if library is None else None
        self.lib = ct.CDLL(str(path))
        self.lib.ir_decoder_abi.restype = ct.c_uint32
        if self.lib.ir_decoder_abi() != 1:
            raise RuntimeError('MPP/RGA codec ABI mismatch')
        self.lib.ir_decoder_error.restype = ct.c_char_p
        self.lib.ir_decoder_build_info.restype = ct.c_char_p
        self.lib.ir_decoder_create.argtypes = [ct.POINTER(ct.c_void_p)]
        self.lib.ir_decoder_decode.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(FrameInfo)]
        if self.profiling:
            if not hasattr(self.lib, 'ir_decoder_decode_profiled'):
                raise RuntimeError('Diagnostic MPP decoder symbol missing; rebuild native codec')
            self.lib.ir_decoder_decode_profiled.argtypes = self.lib.ir_decoder_decode.argtypes+[ct.POINTER(DecodeProfile)]
        self.lib.ir_decoder_crop_bgr.argtypes = [ct.c_void_p]+[ct.c_int]*4+[
            ct.c_void_p, ct.c_size_t, ct.POINTER(CropStats)]
        self.lib.ir_decoder_release_frame.argtypes = [ct.c_void_p]
        self.lib.ir_decoder_destroy.argtypes = [ct.c_void_p]
        self.lib.ir_decoder_release_frame.restype = None
        self.lib.ir_decoder_destroy.restype = None
        self.identity = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             build=self.lib.ir_decoder_build_info().decode(),
                             manifest_verified=build_manifest is not None)
        self.check(self.lib.ir_decoder_create(ct.byref(self.handle)))

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.ir_decoder_error().decode('utf-8', errors='replace'))

    def decode(self, data):
        if self.closed:
            raise RuntimeError('JPEG decoder is closed')
        if self.active is not None:
            raise RuntimeError('Release the previous MPP frame before decoding another')
        started = time.perf_counter_ns()
        if self.backend == 'cpu':
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid encoded image')
            return BGRFrame(image, (time.perf_counter_ns()-started)/1e6)
        if not isinstance(data, bytes):
            data = bytes(data)
        if not data.startswith(b'\xff\xd8'):
            raise ValueError('MPP requires JPEG bytes; PNG/other formats are unsupported')
        info = FrameInfo()
        # c_char_p owns the immutable packet for the complete synchronous decode call.
        packet = ct.c_char_p(data)
        profile = DecodeProfile() if self.profiling else None
        if profile is None:
            self.check(self.lib.ir_decoder_decode(self.handle, packet, len(data), ct.byref(info)))
        else:
            self.check(self.lib.ir_decoder_decode_profiled(self.handle, packet, len(data), ct.byref(info), ct.byref(profile)))
        self.active = HardwareFrame(self, info, (time.perf_counter_ns()-started)/1e6)
        if profile is not None:
            stamps = list(profile.boundary_ns)
            if not all(stamps) or stamps != sorted(stamps):
                self.active.close()
                raise RuntimeError('Invalid native decode timing boundaries')
            self.active.metrics['decode_profile_ns'] = dict(zip(DECODE_BOUNDARIES, stamps))
        return self.active

    def close(self):
        if self.active is not None:
            self.active.close()
        if self.handle.value:
            self.lib.ir_decoder_destroy(self.handle)
            self.handle = ct.c_void_p()
        self.closed = True
