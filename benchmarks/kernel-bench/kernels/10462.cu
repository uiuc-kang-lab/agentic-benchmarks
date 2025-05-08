#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_warp_atomic_kernel(const float* __restrict__ input,
                                        float* output,
                                        float* segment_sums,
                                        int inner_size,
                                        int stride) {
    int idx = blockIdx.x;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    const int warp_size = 32;
    const unsigned mask = 0xffffffff;
    const int lane = threadIdx.x;
    
    // Calculate number of full warps needed for stride dimension
    int num_warps = (stride + warp_size - 1) / warp_size;
    int warp_id = threadIdx.x / warp_size;
    int local_id = lane % warp_size;
    
    // Process stride dimension in warp-sized segments
    float running_sum = 0.0f;
    
    for (int w = 0; w < num_warps; w++) {
        int s = w * warp_size + local_id;
        float val = 0.0f;
        
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            val = __ldg(&input[global_idx]);
        }
        
        // Warp-level inclusive scan
        #pragma unroll
        for (int offset = 1; offset < warp_size; offset *= 2) {
            float n = __shfl_up_sync(mask, val, offset);
            if (local_id >= offset) val += n;
        }
        
        // Only the last thread in warp updates segment sum using atomic
        if (local_id == warp_size - 1 && s < stride) {
            int segment_idx = outer_idx * inner_size + inner_idx;
            if (w > 0) {
                running_sum = atomicAdd(&segment_sums[segment_idx], val);
            } else {
                segment_sums[segment_idx] = val;
                running_sum = 0.0f;
            }
        }
        
        // Broadcast running sum to all threads in warp
        running_sum = __shfl_sync(mask, running_sum, warp_size - 1);
        
        // Add running sum and write result
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            output[global_idx] = val + running_sum;
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
    
    // Allocate temporary storage for segment sums
    auto segment_sums = torch::zeros({outer_size * inner_size}, 
                                   x.options().dtype(torch::kFloat32));
    
    int total_blocks = outer_size * inner_size;
    int threads_per_block = 128; // Use 4 warps per block
    
    cumsum_warp_atomic_kernel<<<total_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        segment_sums.data_ptr<float>(),
        inner_size,
        stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level atomic optimized CUDA cumulative sum");
}