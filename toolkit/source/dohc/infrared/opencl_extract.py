"""Mali GPU fixed-point Gaussian, 8-connected components and integer moments.

This module deliberately imports neither OpenCV nor SciPy so it can run under
the board's system Python, whose OpenCL ICD is compatible with Mali.
"""
from collections import namedtuple
import numpy as np
from .profiling import span, timed

PROFILE_KERNELS = ('horizontal', 'vertical', 'init', 'join', 'compress', 'compact', 'moments',
                   'grayscale', 'mean_horizontal', 'contrast', 'binary', 'morph', 'crop',
                   'masked_copy', 'gen2_signal', 'maximum')

Candidate = namedtuple("Candidate", "center peak core_area")

SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
int reflect(int x, int n) { return x < 0 ? -x : (x >= n ? 2*n-2-x : x); }
__kernel void gen2_signal(__global const uchar *bgr, __global uchar *raw,
    __global uchar *weights, int n, int gray) {
    int i=get_global_id(0); if(i>=n)return;
    int b=bgr[3*i],g=bgr[3*i+1],r=bgr[3*i+2],v=min(b,r);
    // OpenCV uint8 BGR2GRAY uses 15-bit coefficients (14-bit differs by 1).
    v=gray ? (3735*b+19235*g+9798*r+16384)>>15 : (v-g>25 ? v : 0);
    raw[i]=v; weights[i]=v;
}
__kernel void maximum(__global const uchar *a, __global int *out, int n) {
    int start=get_global_id(0)*256,value=0;
    for(int i=start;i<min(start+256,n);i++)value=max(value,(int)a[i]);
    atomic_max(out,value);
}
__kernel void grayscale(__global const uchar *bgr, __global uchar *gray, int n) {
    int i=get_global_id(0); if(i<n)
        gray[i]=(1868*(int)bgr[3*i]+9617*(int)bgr[3*i+1]+4899*(int)bgr[3*i+2]+8192)>>14;
}
__kernel void mean_horizontal(__global const uchar *a, __global ushort *b, int w, int h) {
    int i=get_global_id(0); if(i>=w*h)return;
    int x=i%w,y=i/w,sum=0;
    for(int k=-7;k<=7;k++)sum+=a[y*w+reflect(x+k,w)]; b[i]=sum;
}
__kernel void contrast(__global const uchar *a, __global const ushort *b,
    __global uchar *out, int w, int h) {
    int i=get_global_id(0); if(i>=w*h)return;
    int x=i%w,y=i/w,sum=0;
    for(int k=-7;k<=7;k++)sum+=b[reflect(y+k,h)*w+x];
    out[i]=a[i]>5 ? max((int)a[i]-(sum+112)/225,0) : 0;
}
__kernel void binary(__global const uchar *a, __global uchar *b, int n, int threshold) {
    int i=get_global_id(0); if(i<n)b[i]=a[i]>threshold?255:0;
}
__kernel void masked_copy(__global const uchar *a, __global uchar *b,
    __global const uchar *tiles, int tw, int w, int n) {
    int i=get_global_id(0); if(i<n)b[i]=tiles[((i/w)/16)*tw+(i%w)/16]?a[i]:0;
}
__kernel void morph(__global const uchar *a, __global uchar *b, int w, int h, int dilate) {
    int i=get_global_id(0); if(i>=w*h)return;
    int x=i%w,y=i/w,value=dilate?0:255;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
        int v=a[clamp(y+dy,0,h-1)*w+clamp(x+dx,0,w-1)];
        value=dilate?max(value,v):min(value,v);
    } b[i]=value;
}
__kernel void crop(__global uchar *a, __global int *edge, int w, int h,
    int x0, int y0, int x1, int y1, int sides) {
    int i=get_global_id(0); if(i>=w*h)return;
    int x=i%w,y=i/w;
    if(x<x0||x>=x1||y<y0||y>=y1){a[i]=0;return;}
    if(((sides&1)&&x==x0)||((sides&2)&&y==y0)||
       ((sides&4)&&x==x1-1)||((sides&8)&&y==y1-1))atomic_max(edge,(int)a[i]);
}
__kernel void horizontal(__global const uchar *a, __global ushort *b,
    __global const uchar *tiles, int tw, int w, int h) {
    int i=get_global_id(0); if(i>=w*h)return;
    int x=i%w, y=i/w, sum=0; int k[5]={1,42,170,42,1};
    if(!tiles[(y/16)*tw+x/16]){b[i]=0;return;}
    for(int t=-2;t<=2;t++)sum+=k[t+2]*a[y*w+reflect(x+t,w)]; b[i]=sum;
}
__kernel void vertical(__global const ushort *a, __global uchar *b,
    __global const uchar *tiles, int tw, int w, int h) {
    int i=get_global_id(0); if(i>=w*h)return;
    int x=i%w, y=i/w, sum=0; int k[5]={1,42,170,42,1};
    if(!tiles[(y/16)*tw+x/16]){b[i]=0;return;}
    for(int t=-2;t<=2;t++)sum+=k[t+2]*a[reflect(y+t,h)*w+x]; b[i]=(sum+32768)>>16;
}
__kernel void init(__global const uchar *a, __global int *p, int n, int t) {
    int i=get_global_id(0); if(i<n)p[i]=a[i]>t?i:-1;
}
int root(__global int *p, int i) {
    int j=atomic_add(p+i,0);
    while(j!=i){i=j;j=atomic_add(p+i,0);}return i;
}
__kernel void join(__global int *p, __global int *changed,
    __global const uchar *smooth, int threshold, int w, int h) {
    int i=get_global_id(0);if(i>=w*h||smooth[i]<=threshold)return;
    int x=i%w,y=i/w;
    for(int dy=-1;dy<=0;dy++)for(int dx=-1;dx<=1;dx++) {
        if(dy==0&&dx>=0)continue;
        int xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0)continue;
        int j=yy*w+xx;if(smooth[j]<=threshold)continue;
        int a=root(p,i),b=root(p,j);if(a==b)continue;
        int high=max(a,b),low=min(a,b);
        int previous=atomic_min(p+high,low);
        if(previous>low)atomic_or(changed,1);
    }
}
__kernel void compress(__global int *p, __global const uchar *smooth, int threshold, int n) {
    int i=get_global_id(0);if(i>=n||smooth[i]<=threshold)return;
    int parent=atomic_add(p+i,0);if(parent==i)return;
    int r=root(p,parent);if(parent!=r)atomic_xchg(p+i,r);
}
__kernel void compact(__global const int *p, __global int *map, __global int *count, int n) {
    int i=get_global_id(0);if(i<n&&p[i]==i)map[i]=atomic_inc(count);
}
__kernel void moments(__global const int *p, __global const int *map,
    __global const uchar *gray, __global const uchar *weights, __global int *area, __global int *peak,
    __global int *first, __global ulong *total, __global ulong *sx, __global ulong *sy,
    int n, int w, int threshold) {
    int i=get_global_id(0);if(i>=n||p[i]<0)return;
    int slot=map[p[i]],x=i%w,y=i/w,value=weights[i],weight=max((int)value-threshold+1,0);
    atomic_inc(area+slot);atomic_max(peak+slot,value);
    atomic_min(first+slot,(y/2)*((w+1)/2)+x/2);
    atom_add(total+slot,(ulong)weight);
    atom_add(sx+slot,(ulong)x*(ulong)weight);atom_add(sy+slot,(ulong)y*(ulong)weight);
}
"""


class OpenCLExtractor:
    name = "mali_opencl"
    is_gpu = True

    def __init__(self, npu=None):
        # PyOpenCL retries unwritable cache locks for a minute before failing.
        import os
        from pathlib import Path
        import tempfile
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache")))
        try:
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=str(cache)):
                pass
        except OSError as exc:
            raise RuntimeError("OpenCL cache is not writable: %s (%s)" % (cache, exc)) from exc
        import pyopencl as cl
        devices = [d for p in cl.get_platforms() for d in p.get_devices()
                   if d.type & cl.device_type.GPU]
        if not devices:
            raise RuntimeError("No OpenCL GPU; CPU OpenCL devices are not GPU acceleration")
        self.cl = cl
        self.device = devices[0].name
        self.context = cl.Context([devices[0]])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.program = cl.Program(self.context, SOURCE).build()
        self.kernels = {name: cl.Kernel(self.program, name) for name in
                        ("horizontal", "vertical", "init", "join", "compress", "compact", "moments",
                         "grayscale", "mean_horizontal", "contrast", "binary", "morph", "crop",
                         "masked_copy", "gen2_signal", "maximum")}
        self.shape = None
        self.npu = npu
        self.is_npu = npu is not None
        self.gpu_calls = 0
        self.gpu_ms = 0.
        self.gpu_join_ms = self.gpu_compress_ms = 0.
        self.ccl_rounds = self.control_download_bytes = self.component_download_bytes = 0
        self.events = []
        self.component_capacity = 0
        self.region_buffers = []
        self.regions = []
        self.image_upload_bytes = self.plane_download_bytes = self.preprocess_pixels = 0
        self.name = "rk3588_npu+mali_opencl" if npu else "mali_opencl"

    def stats(self):
        if self.events:
            self.events[-1][1].wait()
        for name, event in self.events:
            elapsed = (event.profile.end-event.profile.start)/1e6
            self.gpu_ms += elapsed
            key = 'gpu_kernel_'+name+'_ms'
            setattr(self, key, getattr(self, key, 0.)+elapsed)
            if name in ('join', 'compress'):
                key = 'gpu_'+name+'_ms'
                setattr(self, key, getattr(self, key)+elapsed)
            self.gpu_calls += 1
        self.events.clear()
        return {"gpu_calls": int(self.gpu_calls),
                "npu_calls": int(getattr(self.npu, "calls", 0)),
                "gpu_ms": self.gpu_ms, "npu_ms": float(getattr(self.npu, "elapsed_ms", 0.)),
                "cpu_fallback": 0, "image_upload_bytes": self.image_upload_bytes,
                "plane_download_bytes": self.plane_download_bytes,
                "preprocess_pixels": self.preprocess_pixels,
                **{'gpu_kernel_'+name+'_ms': getattr(self, 'gpu_kernel_'+name+'_ms', 0.)
                   for name in PROFILE_KERNELS},
                **{key: getattr(self, key) for key in ('gpu_join_ms', 'gpu_compress_ms',
                    'ccl_rounds', 'control_download_bytes', 'component_download_bytes')}}

    def buffer(self, size):
        return self.cl.Buffer(self.context, self.cl.mem_flags.READ_WRITE, size=int(size))

    def fill(self, buffer, value, count):
        self.cl.enqueue_fill_buffer(self.queue, buffer, np.int32(value), 0, int(count)*4)

    def run(self, name, *args, size=None):
        event = self.kernels[name](self.queue, (self.n if size is None else size,), None, *args)
        self.events.append((name, event))
        if len(self.events) >= 512:
            self.stats()
        return event

    def begin_frame(self, image, rectangles=None):
        from .image_source import full_bgr
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 3:
            raise ValueError('Gen2 OpenCL requires uint8 BGR at least 3x3')
        from .methods import merge_rectangles
        h, w = image.shape[:2]
        rectangles = [[0, 0, w, h]] if rectangles is None else rectangles
        if not rectangles or any(len(r) != 4 or
                any(not isinstance(v, (int, np.integer)) for v in r) or
                not (0 <= r[0] < r[2] <= w and 0 <= r[1] < r[3] <= h) for r in rectangles):
            raise ValueError('Gen2 ROIs must be nonempty native-image integer rectangles')
        rectangles = merge_rectangles(rectangles)
        self.allocate(image.shape[:2])
        self.regions = []
        transfers, uploads = [], []
        pixels = sum((x1-x0+4)*(y1-y0+4) for x0, y0, x1, y1 in rectangles)
        if pixels < h*w:
            for index, rectangle in enumerate(rectangles):
                patch = region_image(image, rectangle)
                ph, pw = map(np.int32, patch.shape[:2])
                n, tw = int(ph)*int(pw), np.int32((int(pw)+15)//16)
                tiles = int(tw)*((int(ph)+15)//16)
                if index == len(self.region_buffers):
                    self.region_buffers.append(dict(capacity=0, tile_capacity=0))
                buffers = self.region_buffers[index]
                if n > buffers['capacity']:
                    buffers.update({key: self.buffer(n*size) for key, size in
                                    (('source', 3), ('gray', 1), ('tmp', 2), ('smooth', 1))})
                    buffers['capacity'] = n
                if tiles > buffers['tile_capacity']:
                    buffers['tiles'] = self.buffer(tiles)
                    buffers['tile_capacity'] = tiles
                    self.cl.enqueue_fill_buffer(self.queue, buffers['tiles'], np.uint8(1), 0, tiles)
                self.regions.append((rectangle, pw, ph, tw, buffers))
                uploads.append(patch)  # Keep every nonblocking source alive until the batch completes.
                transfers.append(self.cl.enqueue_copy(self.queue, buffers['source'], patch, is_blocking=False))
                self.image_upload_bytes += patch.nbytes
        else:
            if getattr(self, 'source_capacity', 0) < image.size:
                self.source = self.buffer(image.size)
                self.source_capacity = image.size
            uploads.append(np.ascontiguousarray(full_bgr(image)))
            transfers.append(self.cl.enqueue_copy(self.queue, self.source, uploads[-1], is_blocking=False))
            self.image_upload_bytes += uploads[-1].nbytes
            self.cl.enqueue_fill_buffer(self.queue, self.tiles, np.uint8(1), 0, int(self.tw)*self.th)
        # The in-order queue completes every submitted copy before the last one.
        transfers[-1].wait()

    def prepare_gen2(self, branch):
        if branch not in ('ordinary', 'gray'):
            raise ValueError('Weak-light recovery remains on CPU')
        if self.regions:
            self.fill(self.changed, 0, 1)
            for _, w, h, tw, buffers in self.regions:
                n = int(w)*int(h)
                self.run('gen2_signal', buffers['source'], buffers['gray'], buffers['gray'],
                         np.int32(n), np.int32(branch == 'gray'), size=n)
                self.run('horizontal', buffers['gray'], buffers['tmp'], buffers['tiles'], tw, w, h, size=n)
                self.run('vertical', buffers['tmp'], buffers['smooth'], buffers['tiles'], tw, w, h, size=n)
                # Halo values may only increase this bound; no nonempty core is skipped.
                self.run('maximum', buffers['smooth'], self.changed, np.int32(n), size=(n+255)//256)
                self.preprocess_pixels += n
            maximum = np.empty(1, np.int32)
            self.cl.enqueue_copy(self.queue, maximum, self.changed).wait()
            return int(maximum[0])
        self.run('gen2_signal', self.source, self.gray, self.weights, np.int32(self.n), np.int32(branch == 'gray'))
        self.run('horizontal', self.gray, self.tmp, self.tiles, self.tw, self.w, self.h)
        self.run('vertical', self.tmp, self.smooth, self.tiles, self.tw, self.w, self.h)
        self.fill(self.changed, 0, 1)
        self.run('maximum', self.smooth, self.changed, np.int32(self.n), size=(self.n+255)//256)
        self.preprocess_pixels += self.n
        maximum = np.empty(1, np.int32)
        self.cl.enqueue_copy(self.queue, maximum, self.changed).wait()
        return int(maximum[0])

    def download_regions(self, rectangles):
        patches, transfers = [], []
        regions = {tuple(r): (int(w), int(h), buffers) for r, w, h, _, buffers in self.regions}
        for rectangle in rectangles:
            x0, y0, x1, y1 = rectangle
            if self.regions:
                if tuple(rectangle) not in regions:
                    raise ValueError('ROI was not prepared for this frame')
                w, h, buffers = regions[tuple(rectangle)]
                origin, planes_gpu = (2, 2, 0), (buffers['gray'], buffers['smooth'])
            else:
                w, h = int(self.w), int(self.h)
                origin, planes_gpu = (x0, y0, 0), (self.gray, self.smooth)
            planes = []
            for buffer in planes_gpu:
                host = np.empty((y1-y0, x1-x0), np.uint8)
                transfers.append(self.cl.enqueue_copy(self.queue, host, buffer, buffer_origin=origin,
                    host_origin=(0, 0, 0), region=(x1-x0, y1-y0, 1),
                    buffer_pitches=(w, w*h), is_blocking=False))
                self.plane_download_bytes += host.nbytes
                planes.append(host)
            patches.append((rectangle, *planes))
        if transfers:
            transfers[-1].wait()
        return patches

    def close(self):
        if not hasattr(self, 'queue'):
            return
        try:
            self.queue.finish()
        finally:
            self.events.clear()
            # Dropping all owning references also releases context/queue/program.
            for key in ('source', 'pre_buffers', 'component_buffers', 'gray', 'weights', 'tmp',
                        'smooth', 'parent', 'mapping', 'changed', 'tiles', 'kernels', 'program',
                        'queue', 'context', 'regions', 'region_buffers'):
                self.__dict__.pop(key, None)

    def preprocess(self, image, mode):
        if image.dtype != np.uint8 or image.shape != (2160, 3840, 3):
            raise ValueError("GPU preprocessing requires native uint8 BGR")
        if mode not in ("gray_global", "gray_adaptive", "binary_morph"):
            raise ValueError("Unknown GPU preprocessing mode")
        cl = self.cl
        self.n = image.shape[0]*image.shape[1]
        if not hasattr(self, "pre_buffers"):
            self.pre_buffers = [self.buffer(self.n*k) for k in (3, 1, 2, 1, 1)]
        source, gray, sums, signal, temporary = self.pre_buffers
        cl.enqueue_copy(self.queue, source, np.ascontiguousarray(image))
        self.run("grayscale", source, gray, np.int32(self.n))
        w, h = np.int32(3840), np.int32(2160)
        if mode == "gray_adaptive":
            self.run("mean_horizontal", gray, sums, w, h)
            self.run("contrast", gray, sums, signal, w, h)
        elif mode == "binary_morph":
            self.run("binary", gray, signal, np.int32(self.n), np.int32(80))
            self.run("morph", signal, temporary, w, h, np.int32(1))
            self.run("morph", temporary, signal, w, h, np.int32(0))
        else:
            signal = gray
        raw = np.empty((2160, 3840), np.uint8)
        plane = np.empty_like(raw)
        cl.enqueue_copy(self.queue, raw, gray).wait()
        cl.enqueue_copy(self.queue, plane, signal).wait()
        return raw, plane

    def allocate(self, shape):
        if self.shape != shape:
            self.shape = shape
            self.h, self.w = map(np.int32, shape)
            self.n = int(self.h)*int(self.w)
            self.gray = self.buffer(self.n)
            self.weights = self.buffer(self.n)
            self.tmp = self.buffer(self.n*2)
            self.smooth = self.buffer(self.n)
            self.parent = self.buffer(self.n*4)
            self.mapping = self.buffer(self.n*4)
            self.changed = self.buffer(4)
            self.tw = np.int32((int(self.w)+15)//16)
            self.th = (int(self.h)+15)//16
            self.tiles = self.buffer(int(self.tw)*self.th)

    def prepare(self, gray, weights=None, use_npu=True, tile_map=None, core=None, binary=False):
        if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 3:
            raise ValueError("OpenCL extraction requires a uint8 plane at least 3x3")
        cl = self.cl
        self.allocate(gray.shape)
        self.n = int(gray.size)
        if tile_map is None:
            tile_map = np.ones((self.th, int(self.tw)), np.uint8)
        if self.npu is not None and use_npu:
            # Gaussian weights are positive and sum to one. No pixel with all
            # neighbors <=10 can survive any threshold in the frozen G sweep.
            # One tile halo covers both separable filter passes at native pixels.
            active = self.npu.maxima(gray) > 10
            padded = np.pad(active, 1)
            tile_map = np.zeros_like(active, dtype=np.uint8)
            for y in range(3):
                for x in range(3):
                    tile_map |= padded[y:y+self.th, x:x+int(self.tw)]
        cl.enqueue_copy(self.queue, self.tiles, tile_map)
        cl.enqueue_copy(self.queue, self.gray, np.ascontiguousarray(gray))
        weight_image = gray if weights is None else weights
        if weight_image.dtype != np.uint8 or weight_image.shape != gray.shape:
            raise ValueError("GPU weight plane must match the uint8 signal")
        cl.enqueue_copy(self.queue, self.weights, np.ascontiguousarray(weight_image))
        if binary:
            self.run("masked_copy", self.gray, self.smooth, self.tiles, self.tw, self.w, np.int32(self.n))
        else:
            self.run("horizontal", self.gray, self.tmp, self.tiles, self.tw, self.w, self.h)
            self.run("vertical", self.tmp, self.smooth, self.tiles, self.tw, self.w, self.h)
        self.edge_peak = 0
        if core is not None:
            self.fill(self.changed, 0, 1)
            self.run("crop", self.smooth, self.changed, self.w, self.h,
                     *[np.int32(v) for v in core])
            edge = np.empty(1, np.int32)
            cl.enqueue_copy(self.queue, edge, self.changed).wait()
            self.edge_peak = int(edge[0])

    @timed('gpu_ccl_extract')
    def extract(self, threshold, min_area=2):
        if self.npu is not None and threshold < 10:
            raise ValueError("NPU tile culling is only valid at thresholds >=10")
        cl = self.cl
        flag = np.empty(1, np.int32)
        self.run("init", self.smooth, self.parent, np.int32(self.n), np.int32(threshold))
        for _ in range(128):
            self.fill(self.changed, 0, 1)
            self.run("join", self.parent, self.changed, self.smooth, np.int32(threshold), self.w, self.h)
            self.run("compress", self.parent, self.smooth, np.int32(threshold), np.int32(self.n))
            with span('gpu_ccl_convergence_readback'):
                cl.enqueue_copy(self.queue, flag, self.changed, is_blocking=False).wait()
            self.ccl_rounds += 1
            self.control_download_bytes += flag.nbytes
            if not flag[0]:
                break
        else:
            raise RuntimeError("GPU connected components did not converge; no partial detections")
        self.fill(self.changed, 0, 1)
        self.run("compact", self.parent, self.mapping, self.changed, np.int32(self.n))
        with span('gpu_compact_count_readback'):
            cl.enqueue_copy(self.queue, flag, self.changed, is_blocking=False).wait()
        self.control_download_bytes += flag.nbytes
        count = int(flag[0])
        if count == 0:
            return []
        if count > 262144:
            raise RuntimeError("GPU component storage budget exceeded; no CPU fallback")
        if count > self.component_capacity:
            self.component_buffers = [self.buffer(count*size) for size in (4, 4, 4, 8, 8, 8)]
            self.component_capacity = count
        area, peak, first, total, sx, sy = self.component_buffers
        for buffer in (area, peak):
            self.fill(buffer, 0, count)
        self.fill(first, np.iinfo(np.int32).max, count)
        for buffer in (total, sx, sy):
            self.fill(buffer, 0, count*2)
        self.run("moments", self.parent, self.mapping, self.gray, self.weights, area, peak, first,
                 total, sx, sy, np.int32(self.n), self.w, np.int32(threshold))
        with span('gpu_component_readback'):
            arrays, transfers = [], []
            for buffer, dtype in zip((area, peak, first, total, sx, sy),
                                      (np.int32,)*3+(np.uint64,)*3):
                host = np.empty(count, dtype)
                arrays.append(host)
                transfers.append(cl.enqueue_copy(self.queue, host, buffer, is_blocking=False))
                self.component_download_bytes += host.nbytes
            # Keep hosts and NannyEvents alive through the final in-order transfer.
            transfers[-1].wait()
        area, peak, first, total, sx, sy = arrays
        keep = np.flatnonzero((area >= min_area) & (total > 0))
        if len(keep) > 2048:
            return [Candidate(np.zeros(2, np.float32), 0, 0)]*2049
        keep = keep[np.argsort(first[keep], kind="stable")]
        result = [Candidate(np.float32([sx[i]/total[i], sy[i]/total[i]]),
                            int(peak[i]), int(area[i])) for i in keep]
        result.sort(key=lambda c: (c.peak, c.core_area), reverse=True)
        return result


def region_image(image, rectangle):
    """Native ROI with a two-pixel Gaussian halo, reflected only at image edges."""
    x0, y0, x1, y1 = rectangle
    h, w = image.shape[:2]
    left, top, right, bottom = max(0, x0-2), max(0, y0-2), min(w, x1+2), min(h, y1+2)
    from .image_source import crop_bgr
    patch = crop_bgr(image, [left, top, right, bottom])
    padding = ((max(0, 2-y0), max(0, y1+2-h)),
               (max(0, 2-x0), max(0, x1+2-w)), (0, 0))
    if any(a or b for a, b in padding):
        patch = np.pad(patch, padding, mode='reflect')
    return np.ascontiguousarray(patch)
