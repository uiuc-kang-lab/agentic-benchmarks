#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to perform warp-level inclusive scan
__device__ float warp_inclusive_scan(float val, unsigned mask) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(mask, val, offset);
        if (threadIdx.x >= offset) {
            val += n;
        }
    }
    return val;
}

// Kernel using warp-level scan to compute cumulative sum
__global__ void cumsum_warp_scan_kernel(const float* __restrict__ input, 
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
        
        val = warp_inclusive_scan(val, mask);
        
        val += warp_total;
        
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            output[global_idx] = val;
        }
        
        int valid_count = (stride - seg * warp_size < warp_size) ? (stride - seg * warp_size) : warp_size;
        float seg_total = __shfl_sync(mask, val, valid_count - 1);
        warp_total = seg_total;
    }
}

// Host function to launch the kernel
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
    int total_blocks = outer_size * inner_size;

    cumsum_warp_scan_kernel<<<total_blocks, 32>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        inner_size, 
        stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level optimized CUDA cumulative sum with device function");
}
