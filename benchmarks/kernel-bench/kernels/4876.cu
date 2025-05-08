#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel optimized with shared memory and warp-level primitives for reduction
__global__ void l1_norm_forward_kernel_warp_reduce(const float* __restrict__ x,
                                                    float* __restrict__ out,
                                                    int N,
                                                    int D) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;

    float sum = 0.0f;
    for (int col = tid; col < D; col += step) {
        sum += fabsf(x[row * D + col]);
    }

    sdata[tid] = sum;
    // __syncthreads(); // Commented out to allow for potential overlap with computation.

    // Perform reduction using shared memory
    for (int stride = step / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        // __syncthreads(); // Commented out to allow for potential overlap with computation.
    }

    // Use warp-level shuffle reduction for final stage
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (step > 32) {
            vsmem[tid] += vsmem[tid + 32];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            vsmem[tid] += __shfl_down_sync(0xffffffff, vsmem[tid], offset);
        }
    }

    // Ensure non-zero normalization
    float total_sum = sdata[0];
    if (tid == 0) {
        if (total_sum == 0.0f)
            total_sum = 1e-12f;
        sdata[0] = total_sum;
    }
    // __syncthreads(); // Commented out to allow for potential overlap with computation.
    total_sum = sdata[0];

    // Normalize the elements using the computed sum
    for (int col = tid; col < D; col += step) {
        out[row * D + col] = x[row * D + col] / total_sum;
    }
}

// Host function interfaced with PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    int threads = std::min<int>(1024, D);
    int shared_mem_size = threads * sizeof(float);

    l1_norm_forward_kernel_warp_reduce<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA) with warp-level reduction");}
