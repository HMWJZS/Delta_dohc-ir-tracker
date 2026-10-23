"""NPU tile maxima cull dark areas; geometry always decides tracker identity."""
import numpy as np
import time


class NPUTiles:
    def __init__(self, model):
        from rknnlite.api import RKNNLite
        self.runtime = RKNNLite()
        try:
            if self.runtime.load_rknn(str(model)) != 0:
                raise RuntimeError("Cannot load infrared tile model")
            if self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                raise RuntimeError("Cannot initialize RK3588 NPU")
        except Exception:
            self.runtime.release()
            raise
        self.calls = 0
        self.elapsed_ms = 0.

    def stats(self):
        return {"npu_calls": int(self.calls)}

    def maxima(self, gray):
        started = time.monotonic()
        if gray.shape != (2160, 3840) or gray.dtype != np.uint8:
            raise ValueError("NPU tile graph requires native uint8 4K input")
        output = self.runtime.inference(inputs=[np.ascontiguousarray(gray[None, :, :, None])],
                                        data_format=["nhwc"])
        if not output or np.asarray(output[0]).size != 135*240:
            raise RuntimeError("Invalid NPU tile output")
        values = np.asarray(output[0]).reshape(135, 240)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 255):
            raise RuntimeError("Non-finite or out-of-range NPU tile maximum")
        self.calls += 1
        self.elapsed_ms += (time.monotonic()-started)*1000
        return values

    def close(self):
        self.runtime.release()
