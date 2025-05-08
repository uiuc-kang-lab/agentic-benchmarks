#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that ensures memory coalescing by using row pointers for aligned global memory access
__global__ void l1_norm_forward_kernel_aligned(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int N,
                                                 int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Create row pointers so that consecutive threads access consecutive elements
    const float* row_in = x + row * D;
    float* row_out = out + row * D;

    float sum = 0.0f;
    
    // Coalesced global read: threads read consecutive elements from the row
    for (int j = tid; j < D; j += stride) {
        sum += fabsf(row_in[j]);
    }
    
    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();

    // Perform warp-level reduction first
    sum = warpReduceSum(sum);
    
    // Write warp's result to shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x / warpSize)) {
        float warp_sum = sdata[tid];
        warp_sum = warpReduceSum(warp_sum);
        if (tid == 0) {
            sdata[0] = warp_sum;
        }
    }
    __syncthreads();

    float norm = sdata[0];
    // Avoid division by zero
    if (tid == 0 && norm == 0.0f) {
        norm = 1e-12f;
        sdata[0] = norm;
    }
    __syncthreads();
    norm = sdata[0];
    
    // Coalesced global write: normalize each element
    for (int j = tid; j < D; j += stride) {
        row_out[j] = row_in[j] / norm;
    }
}

// Host function interfaced with PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    
    // Choose threads per block ensuring it does not exceed the number of columns
    int threads = std::min<int>(1024, D);
    int shared_mem_size = threads * sizeof(float);

    l1_norm_forward_kernel_aligned<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA) with global memory alignment optimization");
}
