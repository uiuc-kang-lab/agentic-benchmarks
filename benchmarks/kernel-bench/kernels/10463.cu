#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float warp_scan(float val, const unsigned mask = 0xffffffff) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(mask, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val;
}

__global__ void cumsum_kernel_optimized(const float* __restrict__ input,
                            float* __restrict__ output,
                            float* __restrict__ warp_sums,
                            const int inner_size,
                            const int stride) {
    const int idx = blockIdx.x;
    const int outer_idx = idx / inner_size;
    const int inner_idx = idx % inner_size;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    
    // Base index for this thread's work
    const int base_idx = outer_idx * stride * inner_size + inner_idx;
    
    // Process elements in chunks of 32 (warp size)
    for (int warp_start = warp_id * 32; warp_start < stride; warp_start += warps_per_block * 32) {
        float val = 0.0f;
        const int pos = warp_start + lane_id;
        
        if (pos < stride) {
            val = input[base_idx + pos * inner_size];
        }
        
        // Perform warp-level scan
        val = warp_scan(val);
        
        // Last thread in warp stores sum for next warp
        if (lane_id == 31 && pos < stride) {
            warp_sums[outer_idx * ((stride + 31)/32) + warp_start/32] = val;
        }
        
        // Synchronize only when necessary
        if (warp_start + 32 < stride) {
            __syncthreads();
        }
        
        // Add previous warps' sums
        if (pos < stride && warp_start > 0) {
            float prev_sum = 0.0f;
            #pragma unroll 4
            for (int w = 0; w < warp_start / 32; w++) {
                prev_sum += warp_sums[outer_idx * ((stride + 31)/32) + w];
            }
            val += prev_sum;
        }
        
        // Store result
        if (pos < stride) {
            output[base_idx + pos * inner_size] = val;
        }
    }
}

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
    
    // Allocate temporary storage for warp sums
    auto warp_sums = torch::empty({outer_size * ((stride + 31)/32)}, x.options());
    
    const int total_blocks = outer_size * inner_size;
    const int threads_per_block = 256; // Use 8 warps per block
    
    cumsum_kernel_optimized<<<total_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        warp_sums.data_ptr<float>(),
        inner_size,
        stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum with minimal synchronization");
}