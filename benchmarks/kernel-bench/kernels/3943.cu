#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes the Softsign activation for each element
// and simultaneously accumulates a reduction (sum) of the activated values
// using shared memory and warp-level primitives (__shfl_down_sync) for intra-block reductions.
// The final block sums are atomically added into a global reduction value.
// The kernel returns a tuple: (activation tensor, sum reduction tensor).

__global__ void softsign_kernel_fused(const float* __restrict__ x, float* __restrict__ out, float* reduction, int num_elements) {
    extern __shared__ float sdata[];  // Shared memory for block-level reduction
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: compute softsign and accumulate local sum
    for (int i = idx; i < num_elements; i += stride) {
        float val = x[i];
        float res = val / (1.0f + fabsf(val));
        out[i] = res;
        local_sum += res;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp leader writes its sum to shared memory
    int warpId = threadIdx.x / warpSize;
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        sdata[warpId] = local_sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp sums in shared memory
    int numWarps = blockDim.x / warpSize;
    if (threadIdx.x < numWarps) {
        float sum = sdata[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(reduction, sum);
        }
    }
}

// The forward function launches the fused kernel. It returns a tuple: a tensor with the
// Softsign activation applied elementwise, and a tensor containing the sum reduction of the activated values.

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    // Allocate a tensor to hold the reduction sum (1 element).
    auto red = torch::zeros({1}, x.options());

    int num_elements = x.numel();
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    // Shared memory size: one float per warp in the block
    int shared_mem = (threads / 32) * sizeof(float);

    softsign_kernel_fused<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), red.data_ptr<float>(), num_elements
    );

    // Return a tuple: (elementwise Softsign output, reduction sum of outputs)
    return torch::make_tuple(out, red);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Softsign activation with reduction (CUDA)");
}
