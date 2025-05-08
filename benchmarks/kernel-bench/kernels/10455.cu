#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Hybrid kernel combining warp-level primitives and unrolling
// This approach leverages warp-synchronous programming to process small to medium
// segments of the stride and uses loop unrolling for larger inputs to minimize the overhead.
__global__ void cumsum_hybrid_kernel(const float* __restrict__ input, float* output,
                                     int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const int warp_size = 32;
    int lane = threadIdx.x % warp_size;  // lane id in warp
    int warp_id = threadIdx.x / warp_size;
    int num_warps = blockDim.x / warp_size;
    float warp_total = 0.0f;
    unsigned mask = 0xffffffff;

    for (int inner_idx = warp_id; inner_idx < inner_size; inner_idx += num_warps) {
        // Process stride in segments
        for (int seg = 0; seg < stride; seg += warp_size) {
            float sum = 0.0f;
            int base = outer_idx * stride * inner_size + inner_idx + seg * inner_size;

            // Allow unrolling when entire segment fits within the warp
            #pragma unroll
            for (int s = 0; s < warp_size && seg + s < stride; ++s) {
                int idx = base + s * inner_size;
                float val = (seg + s < stride) ? __ldg(&input[idx]) : 0.0f;

                // Warp scan with inclusive shfl_up
                for (int offset = 1; offset < warp_size; offset *= 2) {
                    float n = __shfl_up_sync(mask, val, offset);
                    if (lane >= offset) val += n;
                }

                // Add previous warp total
                val += warp_total;

                if (seg + s < stride) output[idx] = val;

                // Update with last valid value
                if (s == warp_size - 1 || seg + s + 1 == stride) {
                    warp_total = __shfl_sync(mask, val, lane == warp_size - 1 ? lane : warp_size - 1);
                }
            }
        }
    }
}

// Host function to launch the combined kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    int threads = 256;  // Optimal block size using entire warp for performance stability

    cumsum_hybrid_kernel<<<outer_size, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid CUDA cumulative sum combining warp and unroll optimization");
}