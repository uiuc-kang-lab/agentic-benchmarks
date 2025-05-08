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

__global__ void cumsum_kernel(const float* __restrict__ input,
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
    
    const int base_idx = outer_idx * stride * inner_size + inner_idx;
    
    for (int warp_start = warp_id * 32; warp_start < stride; warp_start += warps_per_block * 32) {
        float val = 0.0f;
        const int pos = warp_start + lane_id;
        
        if (pos < stride) {
            val = input[base_idx + pos * inner_size];
        }
        
        val = warp_scan(val);
        
        if (lane_id == 31 && pos < stride) {
            warp_sums[outer_idx * ((stride + 31)/32) + warp_start/32] = val;
        }
        
        __syncthreads();
        
        if (pos < stride && warp_start > 0) {
            float prev_sum = 0.0f;
            #pragma unroll 4
            for (int w = 0; w < warp_start/32; w++) {
                prev_sum += warp_sums[outer_idx * ((stride + 31)/32) + w];
            }
            val += prev_sum;
        }
        
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
    
    auto warp_sums = torch::empty({outer_size * ((stride + 31)/32)}, x.options());
    
    const int total_blocks = outer_size * inner_size;
    const int threads_per_block = 256;
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cumsum_kernel<<<total_blocks / 2, threads_per_block, 0, stream1>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        warp_sums.data_ptr<float>(),
        inner_size,
        stride
    );
    
    cumsum_kernel<<<total_blocks - total_blocks / 2, threads_per_block, 0, stream2>>>(
        x.data_ptr<float>() + (total_blocks / 2) * stride * inner_size,
        output.data_ptr<float>() + (total_blocks / 2) * stride * inner_size,
        warp_sums.data_ptr<float>() + (total_blocks / 2) * ((stride + 31)/32),
        inner_size,
        stride
    );

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum with stream overlap");
}