#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel assigns one output element to each warp. All 32 threads in a warp collaboratively
// compute the dot-product over the L dimension using warp-level reduction, ensuring uniform control flow.

__global__ void einsum_warp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K,
    int total_outputs
) {
    // Compute a global thread id
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp (group of 32 threads) computes one output element
    int warp_id = global_thread_id / 32;  
    int lane_id = global_thread_id % 32;

    // Ensure that threads in warps beyond our output range become inactive
    if (warp_id >= total_outputs) return;

    // Map the warp_id to the 4D index [b, i, j, k]
    int idx = warp_id;
    int k = idx % K;
    idx /= K;
    int j = idx % J;
    idx /= J;
    int i = idx % I;
    int b = idx / I;

    float sum = 0.0f;
    // Each thread in the warp processes a strided portion of the L dimension
    for (int l = lane_id; l < L; l += 32) {
        int a_offset = b * (I * J * L) + i * (J * L) + j * L + l;
        int b_offset = l * K + k;
        sum += A[a_offset] * B[b_offset];
    }

    // Perform warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Lane 0 writes the final sum for this output element
    if (lane_id == 0) {
        C[warp_id] = sum;
    }
}


// The forward function prepares and launches the kernel. Each output element is computed by one warp, so
// we launch total_threads = total_outputs * 32 threads in total.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch: A.size(3) must equal B.size(0)");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);
    int total_outputs = BATCH * I * J * K;  // Each output corresponds to one warp

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Determine grid dimensions
    int warps_needed = total_outputs;
    int total_threads = warps_needed * 32;  // 32 threads per warp
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    einsum_warp_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K,
        total_outputs
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with warp-level reduction (CUDA)");
}
