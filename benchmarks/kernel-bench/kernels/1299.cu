#include <torch/extension.h>
#include <cuda_runtime.h>

// Declare B in constant memory. Ensure that B (of size L*K) fits within constant memory limits.
__constant__ float B_const[1024]; // Adjust size as needed

// Kernel: Each thread computes one element of the output tensor C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
__global__ void einsum_kernel(
    const float* __restrict__ A,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Determine k (column index) and j (row index in the tensor's last two dims) from blockIdx.x and blockIdx.y
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // corresponds to matrix column
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // corresponds to tensor's j dimension

    // blockIdx.z indexes both batch and I dimension, combined as: index = b * I + i
    int combined = blockIdx.z;
    int b = combined / I;
    int i = combined % I;

    if (j < J && k < K) {
        float sum = 0.0f;
        // Calculate starting offset for A for the given b, i, j
        int offsetA = b * I * J * L + i * J * L + j * L;
        // Iterate over l dimension
        for (int l = 0; l < L; ++l) {
            // B is read from constant memory using B_const
            sum += A[offsetA + l] * B_const[l * K + k];
        }
        // Calculate output index and store the result
        int offsetC = b * I * J * K + i * J * K + j * K + k;
        C[offsetC] = sum;
    }
}

// Forward function called from Python
// Copies B into constant memory and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    // Copy matrix B from the input tensor to constant memory
    size_t B_size = B.numel() * sizeof(float);
    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), B_size, 0, cudaMemcpyDeviceToDevice);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Choose thread dimensions. Here, 32x8 threads per block (tweak as needed).
    dim3 threads(32, 8);
    dim3 blocks(
        (K + threads.x - 1) / threads.x,
        (J + threads.y - 1) / threads.y,
        BATCH * I
    );

    einsum_kernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), BATCH, I, J, L, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication using constant memory for B (CUDA)");
}
