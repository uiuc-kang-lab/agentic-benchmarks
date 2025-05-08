#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses warp-scan (with shuffle intrinsics) for subarrays with length <= 32.
// It avoids any __syncthreads() because warp-level primitives don't require explicit synchronization.
__global__ void cumsum_kernel_warp(const float* __restrict__ input, float* __restrict__ output, int stride, int total_subarrays) {
    // Each warp processes one subarray
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread / warpSize;
    const int lane = global_thread & (warpSize-1);  // Faster modulo for power of 2

    if (warp_id >= total_subarrays) return;
    const int base = warp_id * stride;

    // Load element if lane is within the subarray length
    register float val = (lane < stride) ? input[base + lane] : 0.0f;

    // Use warp-scan with shuffle intrinsics; no __syncthreads() is needed here
    unsigned full_mask = 0xffffffff;
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(full_mask, val, offset);
        if (lane >= offset && lane < stride) {
            val += n;
        }
    }

    if (lane < stride) {
        output[base + lane] = val;
    }
}

// For larger stride values, fall back to a sequential accumulation per subarray.
// Each thread processes one subarray independently, so no synchronization is needed.
__global__ void cumsum_kernel_seq(const float* input, float* output, int stride, int total_subarrays) {
    int subarray = blockIdx.x * blockDim.x + threadIdx.x;
    if (subarray >= total_subarrays) return;
    int base = subarray * stride;
    float sum = 0.0f;
    for (int i = 0; i < stride; ++i) {
        sum += input[base + i];
        output[base + i] = sum;
    }
}

// The forward function selects the appropriate kernel based on the stride size.
// For small strides (<= 32), we use the warp-scan kernel which avoids excessive synchronizations.
// For larger strides, a sequential per-subarray accumulation is used.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;
    
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }
    
    int stride = x.size(dim);
    int total_subarrays = outer_size * inner_size;

    if (stride <= 32) {
        // Use warp-scan: each subarray is handled by one warp (32 threads)
        int total_threads = total_subarrays * 32;
        int threads_per_block = 128;  // Must be a multiple of 32
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        cumsum_kernel_warp<<<blocks, threads_per_block>>>(
            x.data_ptr<float>(), 
            output.data_ptr<float>(), 
            stride, 
            total_subarrays
        );
    } else {
        // For larger strides, use the sequential kernel where each thread processes one subarray
        int threads_per_block = 256;
        int blocks = (total_subarrays + threads_per_block - 1) / threads_per_block;
        cumsum_kernel_seq<<<blocks, threads_per_block>>>(
            x.data_ptr<float>(), 
            output.data_ptr<float>(), 
            stride, 
            total_subarrays
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum hybrid using warp-scan for small strides");
}
