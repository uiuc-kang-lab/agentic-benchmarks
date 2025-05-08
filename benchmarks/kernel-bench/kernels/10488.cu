#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256  // Must be multiple of WARP_SIZE

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_aligned_cumsum_kernel(const float* input, float* output, 
                                         const int stride, const int inner_size) {
    extern __shared__ float s_data[];
    
    const int line_idx = blockIdx.x;
    const int outer_idx = line_idx / inner_size;
    const int inner_idx = line_idx % inner_size;
    
    // Base pointers for this line
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Compute warp-aligned chunks
    const int elements_per_warp = (stride + num_warps - 1) / num_warps;
    const int warp_chunk_size = ((elements_per_warp + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int warp_start = warp_id * warp_chunk_size;
    const int thread_start = warp_start + lane_id;
    
    // First pass: compute warp-level partial sums
    float thread_sum = 0.0f; 
    if (thread_start < stride) { 
        thread_sum += in_line[thread_start * inner_size]; 
    }
    const int warp_end = min(warp_start + warp_chunk_size, stride);
    
    // Each thread processes elements strided by WARP_SIZE
    for (int i = thread_start; i < warp_end; i += WARP_SIZE) {
        if (i < stride) {
            thread_sum += in_line[i * inner_size];
        }
    }
    
    // Compute warp-level sum
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results in shared memory
    if (lane_id == 0) {
        s_data[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Compute prefix sum across warps
    if (warp_id == 0 && lane_id < num_warps) {
        float warp_prefix = s_data[lane_id];
        #pragma unroll
        for (int offset = 1; offset < num_warps; offset *= 2) {
            float temp = __shfl_up_sync(0xffffffff, warp_prefix, offset);
            if (lane_id >= offset) {
                warp_prefix += temp;
            }
        }
        s_data[lane_id] = warp_prefix;
    }
    __syncthreads();
    
    // Compute final output with correct offsets
    float warp_offset = (warp_id == 0) ? 0.0f : s_data[warp_id - 1];
    float running_sum = warp_offset;
    
    // Each thread processes its assigned elements
    for (int i = thread_start; i < warp_end; i += WARP_SIZE) {
        if (i < stride) {
            float val = in_line[i * inner_size];
            running_sum += val;
            out_line[i * inner_size] = running_sum;
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
    
    // Launch kernel with warp-aligned blocks
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    warp_aligned_cumsum_kernel<<<total_lines, BLOCK_SIZE, num_warps * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned CUDA cumulative sum");
}