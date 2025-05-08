#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 256;
const int STRIDE_THRESHOLD = 512;  // Threshold to choose between kernels

__global__ void small_stride_kernel(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int outer_size, 
                                  int inner_size, 
                                  int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        const float* in_ptr = input + outer_idx * stride * inner_size + inner_idx;
        float* out_ptr = output + outer_idx * stride * inner_size + inner_idx;
        
        #pragma unroll 4
        for (int i = 0; i < stride; ++i) {
            sum += __ldg(in_ptr + i * inner_size);
            out_ptr[i * inner_size] = sum;
        }
    }
}

__global__ void large_stride_kernel(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int stride, 
                                  int inner_size) {
    int line_index = blockIdx.x;
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;
    
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;
    
    int tid = threadIdx.x;
    int chunk_size = (stride + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, stride);

    extern __shared__ float sdata[];
    
    float thread_sum = 0.0f;
    #pragma unroll 2
    for (int i = start; i < end; i++) {
        thread_sum += __ldg(&in_line[i * inner_size]);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Perform warp-level parallel prefix sum first
    unsigned int lane_id = threadIdx.x % warpSize;
    float warp_sum = thread_sum;
    
    #pragma unroll
    for (int offset = 1; offset < warpSize; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, warp_sum, offset);
        if (lane_id >= offset) warp_sum += n;
    }
    
    // Store the warp results and synchronize
    if (lane_id == warpSize-1) sdata[tid/warpSize] = warp_sum;
    __syncthreads();
    
    // Process the block-level sum using the first warp
    if (tid < warpSize) {
        float val = (tid < (blockDim.x/warpSize)) ? sdata[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = 1; offset < warpSize; offset *= 2) {
            float n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane_id >= offset) val += n;
        }
        sdata[tid] = val;
    }
    __syncthreads();
    
    // Compute final sum for each thread
    float add_offset = 0.0f;
    if (tid > 0) {
        add_offset = sdata[(tid/warpSize)-1];
        add_offset += warp_sum - thread_sum;
    }
    
    float local_running = 0.0f;
    for (int i = start; i < end; i++) {
        local_running += __ldg(&in_line[i * inner_size]);
        out_line[i * inner_size] = local_running + add_offset;
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= x.size(i);
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= x.size(i);
    
    int stride = x.size(dim);
    
    if (stride <= STRIDE_THRESHOLD) {
        dim3 grid(outer_size, (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        small_stride_kernel<<<grid, BLOCK_SIZE>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), 
            outer_size, inner_size, stride);
    } else {
        int total_lines = outer_size * inner_size;
        large_stride_kernel<<<total_lines, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), 
            stride, inner_size);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid CUDA cumulative sum implementation");
}