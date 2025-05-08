#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// Define block size and maximum number of elements to use constant memory for the input
#define BLOCK_SIZE 256
#define CONST_MAX_SIZE 16384  // 16384 * sizeof(float) = 65536 bytes (64KB)

// __constant__ memory declarations
__constant__ float d_norm;                       // Stores computed norm
__constant__ float d_input_const[CONST_MAX_SIZE];  // Stores input tensor for small tensors

// -------------------------------------------------------------------------
// Global memory kernels (for large tensors)
// -------------------------------------------------------------------------

// Kernel to compute sum of squares from global memory input
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float val = input[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// Kernel to normalize using global memory input; reads computed norm from constant memory
__global__ void normalize_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / d_norm;
    }
}

// -------------------------------------------------------------------------
// Constant memory kernels (for small tensors)
// -------------------------------------------------------------------------

// Kernel to compute sum of squares while reading input from constant memory
__global__ void compute_norm_const_kernel(float* norm_out, int numel) {
    __shared__ float shared_sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float val = d_input_const[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// Kernel to normalize using input stored in constant memory
__global__ void normalize_const_kernel(float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = d_input_const[idx] / d_norm;
    }
}

// Host function that selects the appropriate kernel based on tensor size
torch::Tensor forward(torch::Tensor input) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    int numel = input.numel();
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());
    float* norm_ptr = norm_tensor.data_ptr<float>();

    const int threads = BLOCK_SIZE;
    const int blocks = std::min(65535, (numel + threads - 1) / threads);

    if (numel <= CONST_MAX_SIZE) {
        // For small tensors, copy the entire input into constant memory
        const float* input_ptr = input.data_ptr<float>();
        cudaMemcpyToSymbol(d_input_const, input_ptr, numel * sizeof(float), 0, cudaMemcpyDeviceToDevice);

        // Launch kernel that reads input from constant memory
        compute_norm_const_kernel<<<blocks, threads>>>(norm_ptr, numel);
    } else {
        // For larger tensors, use input from global memory
        const float* input_ptr = input.data_ptr<float>();
        compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    }

    // Retrieve computed sum of squares, take square root to obtain Frobenius norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Copy the computed norm into constant memory for fast access
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float), 0, cudaMemcpyHostToDevice);

    if (numel <= CONST_MAX_SIZE) {
        float* output_ptr = output.data_ptr<float>();
        normalize_const_kernel<<<blocks, threads>>>(output_ptr, numel);
    } else {
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with input constant memory optimization");
}
