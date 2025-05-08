#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void reverse_cumsum_kernel(float* output, const float* input, 
                                    const int size, const int stride) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * ELEMENTS_PER_THREAD * blockDim.x + tid;
    
    // Load data into shared memory in reverse order
    if (gid < size) {
        shared_data[tid] = input[size - 1 - gid];
    } else {
        shared_data[tid] = 0;
    }
    __syncthreads();
    
    // Perform parallel scan in shared memory
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float temp = 0;
        if (tid >= offset) {
            temp = shared_data[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            shared_data[tid] += temp;
        }
        __syncthreads();
    }
    
    // Write results back to global memory in correct order
    if (gid < size) {
        output[size - 1 - gid] = shared_data[tid];
    }
    
    // Handle block boundaries using atomic operations only when necessary
    if (blockIdx.x > 0 && tid == 0) {
        float block_sum = shared_data[blockDim.x - 1];
        atomicAdd(&output[size - 1 - gid + blockDim.x], block_sum);
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    x = x.contiguous();
    
    auto output = at::empty_like(x);
    
    const int size = x.size(dim);
    const int stride = x.stride(dim);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    reverse_cumsum_kernel<<<grid, block, 0, stream>>>(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        size,
        stride
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Reverse cumulative sum (CUDA)");
}