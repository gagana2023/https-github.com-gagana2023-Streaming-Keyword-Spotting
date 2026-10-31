"""ONNX Runtime wrapper that reuses its input and output buffers (IOBinding).

`session.run` allocates a new output array on every call. With IOBinding the input and output
tensors are bound once to preallocated NumPy arrays, so the hot loop only overwrites `input`
in place, calls `run()` and reads `logits`. The binding holds raw pointers into these arrays,
so they must stay alive and keep their address: never rebind `self.input` or `self.logits`,
only write into them.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class ORTModel:
    def __init__(
        self,
        path: Path,
        threads: int = 1,
        n_frames: int = 49,
        n_mfcc: int = 10,
        n_classes: int = 12,
    ) -> None:
        """`threads` is the intra-op thread count; 0 lets ORT use all cores."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = np.zeros((1, 1, n_frames, n_mfcc), dtype=np.float32)
        self.logits = np.zeros((1, n_classes), dtype=np.float32)
        self._binding = self.session.io_binding()
        self._binding.bind_cpu_input(self.session.get_inputs()[0].name, self.input)
        self._binding.bind_output(
            name=self.session.get_outputs()[0].name,
            device_type="cpu",
            device_id=0,
            element_type=np.float32,
            shape=self.logits.shape,
            buffer_ptr=self.logits.ctypes.data,
        )

    @property
    def frames(self) -> NDArray[np.float32]:
        """The (n_frames, n_mfcc) view of the input tensor that the feature buffer fills."""
        view: NDArray[np.float32] = self.input[0, 0]
        return view

    def run(self) -> NDArray[np.float32]:
        """Run on the current contents of `input`; returns the (n_classes,) logits view."""
        self.session.run_with_iobinding(self._binding)
        row: NDArray[np.float32] = self.logits[0]
        return row
