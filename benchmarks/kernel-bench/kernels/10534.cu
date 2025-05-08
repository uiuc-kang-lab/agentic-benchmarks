#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__global__ void cumsum_kernel_warp(const float* __restrict__ input, 
                                 float* __restrict__ output,
                                 int outer_size, int inner_size, int stride) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int outer_idx = blockIdx.x;
    
    if (outer_idx < outer_size) {
        // Process WARP_SIZE elements at a time
        for (int base = warp_id * WARP_SIZE; base < inner_size; base += blockDim.x / WARP_SIZE * WARP_SIZE) {
            int idx = base + lane_id;
            if (idx < inner_size) {
                float sum = 0.0f;
                
                #pragma unroll
                for (int i = 0; i < stride; ++i) {
                    int global_idx = outer_idx * stride * inner_size + i * inner_size + idx;
                    float val = __ldg(&input[global_idx]);
                    
                    // Perform warp-level prefix sum for better efficiency
                    #pragma unroll
                    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                        float n = __shfl_up_sync(0xffffffff, sum, offset);
                        if (lane_id >= offset) {
                            sum += n;
                        }
                    }
                    
                    output[global_idx] = sum;
                    
                    // Synchronize warp before next iteration
                    __syncwarp();
                }
            }
        }
    }
}

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
    
    // Use multiple warps per block for better occupancy
    int threads_per_block = 256; // Multiple of WARP_SIZE
    cumsum_kernel_warp<<<outer_size, threads_per_block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(),
        outer_size, inner_size, stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized CUDA cumulative sum");
}