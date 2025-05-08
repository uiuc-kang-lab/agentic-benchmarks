#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__global__ void cumsum_kernel(const float* __restrict__ input, float* output, 
                            int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Process multiple elements per thread using warp-level parallelism
    for (int inner_base = warp_id * WARP_SIZE; inner_base < inner_size; inner_base += (blockDim.x / WARP_SIZE) * WARP_SIZE) {
        int inner_idx = inner_base + lane_id;
        
        if (inner_idx < inner_size) {
            float sum = 0.0f;
            int base_idx = outer_idx * stride * inner_size + inner_idx;
            
            #pragma unroll 16
            for (int i = 0; i < stride; ++i) {
                int idx = base_idx + i * inner_size;
                float val = __ldg(&input[idx]);
                
                // Use warp-level primitives for efficient communication
                #pragma unroll
                for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                    float n = __shfl_up_sync(0xffffffff, sum, offset);
                    if (lane_id >= offset) {
                        sum += n;
                    }
                }
                
                output[idx] = sum + val;
                sum += val;
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
    const int threads_per_block = 256;  // Multiple of WARP_SIZE
    dim3 grid(outer_size);
    dim3 block(threads_per_block);
    
    cumsum_kernel<<<grid, block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), 
        outer_size, inner_size, stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with warp optimization");
}