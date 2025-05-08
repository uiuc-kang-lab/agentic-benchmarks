#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__device__ inline float warpReduceSum(float val) {
    // Assuming a warp has 32 threads
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel that computes row-wise L1 normalization using warp-level primitives
__global__ void l1_norm_warp_optimized_kernel(const float* __restrict__ x,
                                               float* __restrict__ out,
                                               int N,
                                               int D) {
    // Each block handles one row
    int row = blockIdx.x;
    int row_offset = row * D;
    float local_sum = 0.0f;

    // Process as many elements as possible using 128-bit (float4) vectorized loads
    int nvec = D / 4;  // number of full float4 elements
    int rem = D % 4;   // remaining elements

    // Pointer for vectorized loads; we assume x is contiguous and appropriately aligned
    const float4* x_vec = reinterpret_cast<const float4*>(x + row_offset);

    // Each thread loads consecutive vectorized elements, ensuring coalesced access
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        local_sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    // Process any remaining elements for this row
    int base = nvec * 4;
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        local_sum += fabsf(x[row_offset + base + j]);
    }

    // Perform warp-level reduction
    float warp_sum = warpReduceSum(local_sum);

    // Use warp-level reduction to compute the final sum
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&out[row], warp_sum);
    }
    __syncthreads();

    // Normalize the row
    float norm = out[row];
    if (norm == 0.0f) norm = 1e-12f;  // Avoid division by zero

    // Normalization: use vectorized stores to write results in a coalesced manner
    float4* out_vec = reinterpret_cast<float4*>(out + row_offset);
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        float4 res;
        res.x = v.x / norm;
        res.y = v.y / norm;
        res.z = v.z / norm;
        res.w = v.w / norm;
        out_vec[i] = res;
    }
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        int idx = row_offset + base + j;
        out[idx] = x[idx] / norm;
    }
}

// The forward function that sets up the grid and launches the kernel
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    x = x.contiguous();
    
    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    // Choose number of threads per block - use 256 as a good default
    // that balances occupancy and flexibility
    const int threads = 256;

    // Launch one block per row
    l1_norm_warp_optimized_kernel<<<N, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization with warp-level reduction (CUDA)");
}
