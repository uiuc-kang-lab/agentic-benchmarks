from torch.utils.cpp_extension import load
import time
import os

kernel_id = os.getenv("KERNEL_ID")

def build_kernel(*args, **kwargs):
    cuda_module = load(
        name=f"kernel_module_{int(time.time())}",
        sources=[f"kernels/{kernel_id}.cu"],
        extra_cflags=[""],
        extra_ldflags=[""],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module