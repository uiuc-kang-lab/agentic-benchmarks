#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for computing sum of squares using shared memory and warp-level primitives
__global__ void optimized_compute_norm(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32;

    // Initialize shared memory
    float sum = 0.0f;
    while (idx < numel) {
        sum += input[idx] * input[idx];
        idx += blockDim.x * gridDim.x;
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Use warp reduction with shuffle inside each warp
    sum = shared_sum[tid];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread writes the result
    if (lane == 0) {
        atomicAdd(norm_out, sum);
    }
}

// CUDA kernel for normalization
__global__ void normalize_kernel(const float* input, float* output, 
                               float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor with same shape as input
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());
    
    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();
    
    // Calculate total number of elements
    int numel = input.numel();
    
    // Calculate grid and block dimensions
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // First kernel: compute sum of squares
    optimized_compute_norm<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    // Get norm value and compute square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);
    
    // Second kernel: normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}
