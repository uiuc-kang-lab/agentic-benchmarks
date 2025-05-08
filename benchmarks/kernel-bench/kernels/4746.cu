#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_data[256];
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_sum[tid] = 0.0f;
    
    // Compute partial sums using shared memory
    while (idx < numel) {
        // Load data into shared memory
        shared_data[tid] = input[idx];
        shared_sum[tid] += shared_data[tid] * shared_data[tid];
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
    
    // Reduce within block using shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

__global__ void normalize_kernel(const float* input, float* output, 
                               float norm, int numel) {
    __shared__ float shared_input[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < numel) {
        // Load data into shared memory
        shared_input[tid] = input[idx];
        // Normalize using shared memory
        output[idx] = shared_input[tid] / norm;
        idx += blockDim.x * gridDim.x;
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
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

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