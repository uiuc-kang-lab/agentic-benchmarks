#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float warp_scan(float val) {
    float sum = val;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(FULL_MASK, sum, offset);
        if (threadIdx.x >= offset) {
            sum += n;
        }
    }
    return sum;
}

__global__ void warp_cumsum_kernel(const float* input, float* output, 
                                 int stride, int inner_size) {
    int line_index = blockIdx.x;
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;
    
    // Base pointers for this line
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Each warp handles a section of the stride
    int elements_per_warp = (stride + warps_per_block - 1) / warps_per_block;
    int warp_start = warp_id * elements_per_warp;
    int warp_end = min(warp_start + elements_per_warp, stride);
    
    // Process elements in chunks of WARP_SIZE
    float warp_prefix = 0.0f;
    
    for (int base = warp_start; base < warp_end; base += WARP_SIZE) {
        int idx = base + lane_id;
        float val = (idx < warp_end) ? in_line[idx * inner_size] : 0.0f; if (val != val) val = 0.0f; // Handle NaN values
        
        // Perform warp-level scan
        float scan_result = warp_scan(val);
        
        // Write results with offset from previous iterations
        if (idx < warp_end) {
            out_line[idx * inner_size] = scan_result + warp_prefix;
        }
        
        // Update prefix for next iteration
        warp_prefix += __shfl_sync(FULL_MASK, scan_result, WARP_SIZE-1);
    }
    
    // Propagate final warp sums across warps using the first thread of each warp
    if (lane_id == 0 && warp_id > 0) {
        float warp_sum = warp_prefix;
        int prev_elements = warp_start;
        
        // Update all elements in this warp's section with previous warps' sum
        for (int i = warp_start; i < warp_end; i++) {
            if (i < stride) {
                output[outer_idx * stride * inner_size + i * inner_size + inner_idx] += 
                    __shfl_sync(FULL_MASK, warp_sum, warp_id-1);
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
    int total_lines = outer_size * inner_size;
    
    // Use multiple warps per block for better occupancy
    int threads_per_block = 128; // 4 warps per block
    
    warp_cumsum_kernel<<<total_lines, threads_per_block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), 
        stride, inner_size
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum (warp optimized)");
}