#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Threshold for switching between algorithms
#define STRIDE_THRESHOLD 128
#define INNER_SIZE_THRESHOLD 64

__global__ void cumsum_kernel_small_stride(const float* __restrict__ input, 
                                         float* output,
                                         int outer_size, 
                                         int inner_size, 
                                         int stride) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        int base = outer_idx * stride * inner_size + inner_idx;
        
        #pragma unroll 8
        for (int s = 0; s < stride; s++) {
            int idx = base + s * inner_size;
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

__global__ void cumsum_kernel_large_stride(const float* __restrict__ input, 
                                         float* output,
                                         int inner_size, 
                                         int stride) {
    int idx = blockIdx.x;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    const int warp_size = 32;
    int total_segments = (stride + warp_size - 1) / warp_size;
    int lane = threadIdx.x;
    float warp_total = 0.0f;
    unsigned mask = 0xffffffff;

    for (int seg = 0; seg < total_segments; seg++) {
        int s = seg * warp_size + lane;
        float val = 0.0f;
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            val = input[global_idx];
        }
        
        for (int offset = 1; offset < warp_size; offset *= 2) {
            float n = __shfl_up_sync(mask, val, offset);
            if (lane >= offset) val += n;
        }
        
        val += warp_total;
        
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            output[global_idx] = val;
        }
        
        int valid_count = min(warp_size, stride - seg * warp_size);
        warp_total = __shfl_sync(mask, val, valid_count - 1);
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;
    
    int outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) inner_size *= x.size(i);
    
    int stride = x.size(dim);
    
    // Choose algorithm based on tensor dimensions
    if (stride <= STRIDE_THRESHOLD || inner_size >= INNER_SIZE_THRESHOLD) {
        int threads = std::min(256, inner_size);
        cumsum_kernel_small_stride<<<outer_size, threads>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), 
            outer_size, inner_size, stride
        );
    } else {
        int total_blocks = outer_size * inner_size;
        cumsum_kernel_large_stride<<<total_blocks, 32>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), 
            inner_size, stride
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid adaptive CUDA cumulative sum");
}