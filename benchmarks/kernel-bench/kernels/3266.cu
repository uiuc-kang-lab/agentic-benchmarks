#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel computes the 25_Swish activation and, as an example of replacing shared memory reductions,
// it performs a warp-level reduction of the computed outputs using __shfl_down_sync().
// The warp-level reduction aggregates a partial sum from each warp and writes it to a temporary buffer.
// While the reduction is not used in the final output, it demonstrates how to avoid shared memory overhead
// in cases where small reductions are required.

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ y, float* __restrict__ warpSums, int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Process elements in a grid-stride loop for better occupancy
    for (int i = tid; i < n; i += stride) {
        float val = x[i];
        float sig = 1.0f / (1.0f + expf(-val));
        float res = val * sig;
        y[i] = res;
        local_sum += res;
    }

    // Perform warp-level reduction without shared memory using __shfl_down_sync
    unsigned int mask = 0xffffffff; // All threads in a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Write the warp's reduced sum to global memory from lane 0
    int lane = threadIdx.x & (warpSize - 1);
    if (lane == 0) {
        int warpId = threadIdx.x / warpSize;
        int blockWarpCount = blockDim.x / warpSize;
        int globalWarpId = blockIdx.x * blockWarpCount + warpId;
        warpSums[globalWarpId] = local_sum;
    }
}

// The forward function allocates a temporary buffer for warp sums but returns only the swish activated tensor y.
// The use of grid-stride loops and warp-level reduction minimizes shared memory overhead and can reduce runtime
// in kernels that require small reductions as part of their computation.

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Allocate temporary buffer for warp-level reductions
    int warpsPerBlock = threads / 32;
    int totalWarps = blocks * warpsPerBlock;
    auto warpSums = torch::empty({totalWarps}, x.options());

    swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        warpSums.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with warp-level reduction (CUDA)");
}
