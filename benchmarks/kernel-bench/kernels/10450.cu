#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses warp-level primitives (__shfl_up_sync) to perform an inclusive scan
// along the cumulative sum (cumsum) dimension in segments of warp size (32). Each block
// processes one (outer, inner) pair and divides the stride dimension into segments, eliminating
// the need for shared memory and reducing synchronization overhead.
__global__ void cumsum_warp_scan_kernel(const float* __restrict__ input, 
                                          float* output, 
                                          int inner_size, 
                                          int stride) {
    // Each block corresponds to one (outer, inner) combination
    int idx = blockIdx.x;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    const int warp_size = 32;
    int total_segments = (stride + warp_size - 1) / warp_size; // ceiling division
    int lane = threadIdx.x;  // lane id in the warp (0 to 31)
    float warp_total = 0.0f; // carries the cumulative sum from previous segments
    unsigned mask = 0xffffffff; // full warp mask

    // Process the stride dimension in segments of warp_size
    for (int seg = 0; seg < total_segments; seg++) {
        int s = seg * warp_size + lane; // actual index in the stride dimension
        float val = 0.0f;
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            val = input[global_idx];
        }
        
        // Perform warp-level inclusive scan using __shfl_up_sync
        for (int offset = 1; offset < warp_size; offset *= 2) {
            float n = __shfl_up_sync(mask, val, offset);
            if (lane >= offset) {
                val += n;
            }
        }
        
        // Add the cumulative sum from previous segments
        val += warp_total;
        
        // Write the computed cumsum result to global memory if within bounds
        if (s < stride) {
            int global_idx = outer_idx * (stride * inner_size) + s * inner_size + inner_idx;
            output[global_idx] = val;
        }
        
        // Update warp_total with the last valid value from this segment
        int valid_count = (stride - seg * warp_size < warp_size) ? (stride - seg * warp_size) : warp_size;
        float seg_total = __shfl_sync(mask, val, valid_count - 1);
        warp_total = seg_total;
    }
}

// Host function: sets up dimensions and launches the kernel.
// The cumsum is performed along the 'dim' dimension of the input tensor.
// The tensor is conceptually reshaped into [outer_size, stride, inner_size], and each
// block processes a column identified by (outer_idx, inner_idx).

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
    // Total blocks equals outer_size * inner_size, each block processes one (outer, inner) pair.
    int total_blocks = outer_size * inner_size;

    // Launch kernel with one warp (32 threads) per block
    cumsum_warp_scan_kernel<<<total_blocks, 32>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        inner_size, 
        stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level optimized CUDA cumulative sum");
}
