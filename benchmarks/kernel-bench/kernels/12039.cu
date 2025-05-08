#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// First kernel: compute hinge loss per element and perform intra-block reduction using shared memory and warp-level primitives
__global__ void hinge_loss_reduction_kernel(const float* __restrict__ predictions,
                                              const float* __restrict__ targets,
                                              float* __restrict__ block_sums,
                                              int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Grid-stride loop: each thread accumulates multiple elements
    for (int i = idx; i < n; i += stride) {
        float prod = predictions[i] * targets[i];
        float loss = fmaxf(0.0f, 1.0f - prod);
        sum += loss;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction using shared memory reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction for the final stage
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[tid] = val;
    }

    // Write block's result to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel: reduce block sums to a single sum and compute the mean by dividing with n
__global__ void final_reduction_kernel(const float* __restrict__ block_sums,
                                          float* __restrict__ result,
                                          int num_blocks,
                                          int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread loads multiple block sums (grid-stride if necessary)
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction for final stage
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[tid] = val;
    }

    // Thread 0 writes the mean (sum divided by n) into result
    if (tid == 0) {
        result[0] = sdata[0] / n;
    }
}

// Forward function: launches the two kernels to compute the mean hinge loss
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    auto options = predictions.options();
    
    // Launch configuration for the first kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate temporary tensor for block sums
    auto block_sums = torch::empty({blocks}, options);

    // Launch first kernel with shared memory size of threads * sizeof(float)
    hinge_loss_reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        n
    );

    // Allocate tensor for the final result (a single float)
    auto final_result = torch::empty({1}, options);

    // Use one block for final reduction; each thread will process multiple block sums if needed
    int final_threads = 256;
    final_reduction_kernel<<<1, final_threads, final_threads * sizeof(float)>>>(
        block_sums.data_ptr<float>(),
        final_result.data_ptr<float>(),
        blocks,
        n
    );

    return final_result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with Shared Memory and Warp Reduction");
}
