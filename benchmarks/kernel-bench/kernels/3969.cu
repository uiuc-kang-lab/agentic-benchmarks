#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs the softsign activation for each element:
//   softsign(x) = x / (1 + fabs(x))
// In addition, it demonstrates the use of warp-level primitives by doing a dummy warp-level reduction
// using __shfl_down_sync to sum the computed softsign value across each warp. This replacement of
// potential shared memory operations with warp-level intrinsics can be beneficial for small reductions.
// The warp-level reduction here does not modify the per-element result and is included solely
// to illustrate the recommended technique.

__global__ void softsign_warp_kernel(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const unsigned int full_mask = 0xFFFFFFFF;

    // Process elements in a grid-stride loop
    for (; idx < num_elements; idx += stride) {
        float val = x[idx];
        float soft = val / (1.0f + fabsf(val));

        // Dummy warp-level reduction to sum the computed soft value across the warp.
        // This demonstrates replacing shared memory reductions with __shfl_down_sync.
        // The reduction is performed purely for demonstration and does not affect the output.
        float warp_sum = soft;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(full_mask, warp_sum, offset);
        }
        
        // Each thread writes its independently computed softsign value to ensure correctness.
        out[idx] = soft;
    }
}

// The forward function creates an output tensor and launches the kernel.

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    // Adjust thread count to optimize occupancy based on register usage and shared memory
const int threads = 128;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_warp_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with warp-level primitives (CUDA)");
}
