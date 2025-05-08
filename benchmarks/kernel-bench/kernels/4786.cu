#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel 1: Compute block-wise partial sums
__global__ void compute_partial_sums(const float* input, float* partial_sums, int numel) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_sum[tid] = 0.0f;
    
    while (idx < numel) {
        shared_sum[tid] += input[idx] * input[idx];
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// Kernel 2: Reduce partial sums to final norm
__global__ void reduce_partial_sums(const float* partial_sums, float* norm_out, int num_partials) {
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    shared_sum[tid] = 0.0f;
    int i = tid;
    
    // Grid-stride loop for partial sums
    while (i < num_partials) {
        shared_sum[tid] += partial_sums[i];
        i += blockDim.x;
    }
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// Kernel 3: Normalization
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
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
    
    // Allocate temporary buffer for partial sums
    auto partial_sums = torch::zeros({blocks}, input.options());
    float* partial_sums_ptr = partial_sums.data_ptr<float>();
    
    // Compute partial sums
    compute_partial_sums<<<blocks, threads>>>(input_ptr, partial_sums_ptr, numel);
    
    // Reduce partial sums to final norm
    const int reduce_threads = 256;
    reduce_partial_sums<<<1, reduce_threads>>>(partial_sums_ptr, norm_ptr, blocks);
    
    // Compute final norm and normalize
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val) + 1e-8;  // Prevent division by zero
    
    normalize_kernel<<<(numel + threads - 1) / threads, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm normalization");
}