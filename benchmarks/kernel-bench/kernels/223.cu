#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Block handles a 2D tile of the output matrix
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z; // batch index

    // Each thread block handles a tile of the output matrix
    const int row = by * blockDim.y + ty;
    const int col = bx * blockDim.x + tx;

    if (bz < batch_size && row < M && col < N) {
        float sum = 0.0f;
        // Base pointers for current batch
        const float* batch_A = A + bz * M * K;
        const float* batch_B = B + bz * K * N;

        // Compute dot product
        for (int k = 0; k < K; k++) {
            sum += batch_A[row * K + k] * batch_B[k * N + col];
        }

        // Write result with coalesced access
        C[bz * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Define block and grid dimensions for better memory coalescing
    dim3 threads(16, 16); // 256 threads per block
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y,
        batch_size
    );

    bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication (CUDA)");
}