#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Warp-level reduction using shuffle intrinsics
template <unsigned int WF>
__device__ float warpReduceSum(float val) {
    for (unsigned int offset = WF / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction that reduces values within a block
__device__ float blockReduceSum(float val) {
    // First, reduce within a warp
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    int lane = threadIdx.x & 31;   // lane index within the warp
    int warpId = threadIdx.x >> 5; // warp index within the block

    __shared__ float shared[32];
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // Final reduction within first warp
    if (warpId == 0) {
        val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

// Kernel to compute the sum of squares (Frobenius norm squared) with even workload distribution
__global__ void compute_norm_kernel(const float* __restrict__ input, float* norm_out, int numel) {
    // Evenly partition the input array among blocks
    int totalBlocks = gridDim.x;
    int block_start = (int)(((long long)numel * blockIdx.x) / totalBlocks);
    int block_end   = (int)(((long long)numel * (blockIdx.x + 1)) / totalBlocks);
    int block_len = block_end - block_start;

    // Evenly partition the block's range among threads
    int nthreads = blockDim.x;
    int thread_start = block_start + (block_len * threadIdx.x) / nthreads;
    int thread_end   = block_start + (block_len * (threadIdx.x + 1)) / nthreads;
    int n = thread_end - thread_start;

    float sum = 0.0f;
    const float* base_ptr = input + thread_start;

    // Use vectorized loads if possible, ensuring proper 16-byte alignment, otherwise fallback to scalar loads
    if (((uintptr_t)base_ptr) % 16 == 0) {
        int n4 = n / 4;
        const float4* base_ptr4 = reinterpret_cast<const float4*>(base_ptr);
        for (int i = 0; i < n4; i++) {
            float4 v = base_ptr4[i];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        int rem = n % 4;
        int rem_start = thread_start + n4 * 4;
        for (int i = 0; i < rem; i++) {
            float v = input[rem_start + i];
            sum += v * v;
        }
    } else {
        for (int i = thread_start; i < thread_end; i++) {
            float v = input[i];
            sum += v * v;
        }
    }

    // Perform block-level reduction
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(norm_out, sum);
    }
}

// Constant memory to hold the computed norm
__constant__ float d_norm;

// Kernel to normalize the tensor using the precomputed norm
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
    // Compute number of blocks based on input size; using a similar heuristic as before
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Zero the norm output
    cudaMemset(norm_ptr, 0, sizeof(float));

    // Launch the reduction kernel with even workload partitioning
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Retrieve and compute the final Frobenius norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Copy the computed norm into constant memory for use in normalization
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float), 0, cudaMemcpyHostToDevice);

    // Launch the normalization kernel
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with even workload distribution");
}
