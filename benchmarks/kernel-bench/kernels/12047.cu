#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Compute hinge loss and perform block-level reduction using shared memory and warp-level primitives
__global__ void hinge_loss_reduction_kernel(const float* __restrict__ predictions,
                                              const float* __restrict__ targets,
                                              float* __restrict__ partialSums,
                                              const int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    // Grid-stride loop to accumulate hinge loss values
    for (int i = idx; i < n; i += stride) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        sum += fmaxf(0.0f, 1.0f - pred * targ);
    }

    // Store the partial sum for this thread in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction: reduce in shared memory until we reach warp size (32 threads)
    for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction using __shfl_down_sync, no __syncthreads() needed within a warp
    if (tid < 32) {
        float val = sdata[tid];
        // Unroll warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            partialSums[blockIdx.x] = val;
        }
    }
}

// Forward function which sets up kernel execution and computes the mean hinge loss
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    
    // Set up block and grid dimensions
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Allocate tensor for partial sums computed per block
    auto partialSums = torch::empty({blocks}, predictions.options());

    // Launch kernel with dynamic shared memory size (threads * sizeof(float))
    hinge_loss_reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partialSums.data_ptr<float>(),
        n
    );

    // Final reduction: sum up all block results and compute the mean
    torch::Tensor result = torch::sum(partialSums) / n;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with Shared Memory and Warp-Level Reduction");
}
