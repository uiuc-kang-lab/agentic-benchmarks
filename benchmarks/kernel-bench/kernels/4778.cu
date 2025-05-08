#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Device helper: compute number of iterations for each thread without per-iteration branching
__device__ inline int getIterations(int idx, int stride, int numel) {
    // If idx >= numel, then no valid iteration; otherwise, compute how many iterations can be done
    return (idx < numel) ? ((numel - idx - 1) / stride + 1) : 0;
}

// CUDA kernel to compute the sum of squares (Frobenius norm squared) with minimized warp divergence
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    // Allocate shared memory for storing warp-level partial sums
    __shared__ float warp_sums[32];  // Enough for up to 256 threads => 256/32 = 8 warps

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    // Precompute number of iterations per thread to avoid in-loop branch
    int nIter = getIterations(idx, stride, numel);
    for (int i = 0; i < nIter; i++) {
        int curIdx = idx + i * stride;
        // curIdx is guaranteed < numel
        float val = input[curIdx];
        sum += val * val;
    }

    // Perform warp-level reduction using shuffle instructions to minimize divergence
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp's leader writes its reduced sum to shared memory
    int lane = tid & (warpSize - 1); // equivalent to tid % warpSize
    int warpId = tid >> 5;            // equivalent to tid / warpSize
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // First warp reduces the sums from all warps in this block
    int nw = blockDim.x / warpSize;  // number of warps in this block
    sum = (tid < nw) ? warp_sums[lane] : 0.0f;
    for (int offset = nw / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    if (tid == 0) {
        atomicAdd(norm_out, sum);
    }
}

// CUDA kernel for normalizing the tensor
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Minimal conditional branch to ensure safe memory access
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// C++ forward function called from Python
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
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Launch kernel to compute sum of squares with minimized warp divergence
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Launch kernel to normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Minimized warp divergence Frobenius norm normalization");
}
