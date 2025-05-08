#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_data[BLOCK_SIZE * ELEMENTS_PER_THREAD];
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;
    float thread_sum = 0.0f;
    
    // Load data into shared memory and compute partial sums
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_offset + i * blockDim.x + tid;
        if (idx < numel) {
            float val = input[idx];
            shared_data[tid + i * blockDim.x] = val;
            thread_sum += val * val;
        }
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

__global__ void normalize_kernel(const float* input, float* output, 
                               float norm, int numel) {
    __shared__ float shared_input[BLOCK_SIZE * ELEMENTS_PER_THREAD];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;
    
    // Load data into shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_offset + i * blockDim.x + tid;
        if (idx < numel) {
            shared_input[tid + i * blockDim.x] = input[idx];
        }
    }
    __syncthreads();
    
    // Process and write output
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_offset + i * blockDim.x + tid;
        if (idx < numel) {
            output[idx] = shared_input[tid + i * blockDim.x] / norm;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();
    
    int numel = input.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = min(65535, (numel + (threads * ELEMENTS_PER_THREAD) - 1) / 
                         (threads * ELEMENTS_PER_THREAD));

    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);
    
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}