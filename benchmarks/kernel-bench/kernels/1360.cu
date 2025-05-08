#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the maximum size for the diagonal vector A that can be stored in constant memory
// Adjust MAX_A_SIZE based on hardware limits (typically constant memory is 64KB, so 16384 floats maximum)
#define MAX_A_SIZE 16384

// Constant memory for storing the diagonal vector A
__constant__ float const_A[MAX_A_SIZE];

// Kernel: Each block processes one row of matrix B. The corresponding diagonal element is read from constant memory.
__global__ void diag_matmul_constant_kernel(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;
    if (row < N) {
        // Read the diagonal value from constant memory
        float a_val = const_A[row];
        int row_offset = row * M;
        // Iterate over columns with stride equal to blockDim.x for coalesced access
        for (int col = threadIdx.x; col < M; col += blockDim.x) {
            C[row_offset + col] = a_val * B[row_offset + col];
        }
    }
}

// Forward function that wraps our constant memory based CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous and on the GPU
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Check that N does not exceed the constant memory limit
    TORCH_CHECK(N <= MAX_A_SIZE, "Size of A exceeds the maximum allowed for constant memory");

    // Create an output tensor with the same type and device as B
    auto C = torch::empty({N, M}, B.options());

    // Copy A into constant memory. Since A is already on the device, use device-to-device copy.
    cudaMemcpyToSymbol(const_A, A.data_ptr<float>(), N * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Launch one block per row; assign a fixed number of threads per block for column processing
    const int threads = 256;
    diag_matmul_constant_kernel<<<N, threads>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Create the Pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using constant memory for A on the GPU");
}
