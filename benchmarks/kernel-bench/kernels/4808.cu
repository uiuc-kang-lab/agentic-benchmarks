#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Warp-level reduction without divergent branches using shuffle intrinsics
template <unsigned int WF>
__device__ float warpReduceSum(float val) {
    for (unsigned int offset = WF / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction that minimizes warp divergence by using predicated assignments
__device__ float blockReduceSum(float val) {
    // Perform warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    int lane = threadIdx.x & 31;   // lane index within warp
    int warpId = threadIdx.x >> 5;   // warp index

    __shared__ float warpSums[32];
    // Each warp writes its reduced value from lane 0 into shared memory using a predicated assignment
    float warpVal = __shfl_sync(0xffffffff, val, 0);
    warpSums[warpId] = (lane == 0) ? warpVal : 0.0f;
    __syncthreads();

    // Determine number of warps in the block
    int nWarps = (blockDim.x + 31) >> 5;
    // All threads in the first warp load warp sums, with threads beyond nWarps getting 0
    float sum = (lane < nWarps) ? warpSums[lane] : 0.0f;
    // Final reduction within the first warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

// Device function: Vectorized accumulation of sum of squares using grid-stride loops
__device__ float computeBlockSum(const float* __restrict__ input, int numel) {
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 4 elements at a time using vectorized loads
    int num4 = numel / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    for (int i = idx; i < num4; i += stride) {
        float4 v = input4[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Handle remaining elements
    int start = num4 * 4;
    for (int i = start + idx; i < numel; i += stride) {
        float v = input[i];
        sum += v * v;
    }
    return sum;
}

// Kernel to compute the Frobenius norm (sum of squares) with minimized divergence
__global__ void compute_norm_kernel(const float* __restrict__ input, float* norm_out, int numel) {
    float block_sum = computeBlockSum(input, numel);
    block_sum = blockReduceSum(block_sum);
    if (threadIdx.x == 0) {
        atomicAdd(norm_out, block_sum);
    }
}

// Constant memory variable to store the computed norm
__constant__ float d_norm;

// Kernel to normalize the tensor using the precomputed norm from constant memory
__global__ void normalize_kernel(const float* __restrict__ input, float* output, int numel) {
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

    // Compute the sum of squares (Frobenius norm squared)
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Copy the computed sum from device and compute the square root on host
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Store the final norm in constant memory for fast access
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float), 0, cudaMemcpyHostToDevice);

    // Normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with minimized warp divergence");
}
