#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Warp-level reduction using shuffle intrinsics
template <unsigned int WF>
__device__ float warpReduceSum(float val) {
    for (unsigned int offset = WF/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory and warp-level reduction
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // one element per warp
    int lane = threadIdx.x % 32;   // index within warp
    int wid  = threadIdx.x / 32;   // warp index
    
    // Each warp reduces its own values
    val = warpReduceSum<32>(val);

    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction in the first warp
    if (threadIdx.x < (blockDim.x + 31) / 32) {
        val = shared[lane];
    } else {
        val = 0.0f;
    }
    if (threadIdx.x < 32) {
        val = warpReduceSum<32>(val);
    }
    return val;
}

// Device function to accumulate the sum of squares over a grid-stride loop using vectorized loads
__device__ float computeBlockSum(const float* input, int numel) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vector type for coalesced memory access
    float4* input4 = (float4*)input;
    int numel4 = numel / 4;
    
    // Process 4 elements at a time
    for (int i = idx; i < numel4; i += stride) {
        float4 val4 = input4[i];
        sum += val4.x * val4.x + val4.y * val4.y + 
               val4.z * val4.z + val4.w * val4.w;
    }
    
    // Handle remaining elements
    for (int i = idx + numel4 * 4; i < numel; i += stride) {
        float val = input[i];
        sum += val * val;
    }
    return sum;
}

// Kernel to compute the Frobenius norm (sum of squares) using modular device functions
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    float block_sum = computeBlockSum(input, numel);
    block_sum = blockReduceSum(block_sum);
    if (threadIdx.x == 0) {
        atomicAdd(norm_out, block_sum);
    }
}

// Declare a constant memory variable to store the computed norm
__constant__ float d_norm;

// Kernel to normalize the tensor using the precomputed norm from constant memory
__global__ void normalize_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / d_norm;
    }
}

// Host function that launches the kernels
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
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Launch kernel to compute the sum of squares (Frobenius norm squared)
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Copy the computed sum from device and take the square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Store the final norm in constant memory for fast access
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float), 0, cudaMemcpyHostToDevice);

    // Launch kernel to normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with modular device functions");
}
