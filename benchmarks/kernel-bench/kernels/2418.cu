#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum number of elements that can be stored in constant memory
// For float: 16384 elements (64KB limit), for double: 8192 elements
#define MAX_FLOAT_B_ELEMS 16384
#define MAX_DOUBLE_B_ELEMS 8192

// Declare constant memory for B matrix for float and double
__constant__ float B_const_float[MAX_FLOAT_B_ELEMS];
__constant__ double B_const_double[MAX_DOUBLE_B_ELEMS];

// CUDA kernel for matrix multiplication with transposed A and B, using constant memory for B
template <typename scalar_t>
__global__ void matmul_transpose_const_kernel(
    const scalar_t* __restrict__ A,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            // Access A as transposed (k, M) -> A[k * M + row]
            scalar_t a_val = A[k * M + row];
            scalar_t b_val;
            // Load B from constant memory, where B was stored in transposed form
            if constexpr (std::is_same<scalar_t, float>::value) {
                b_val = B_const_float[col * K + k];
            } else {
                b_val = B_const_double[col * K + k];
            }
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

// Wrapper function that copies B to constant memory and launches the kernel
torch::Tensor matmul_transpose_const_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Ensure A and B have the same scalar type
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same scalar type.");

    // Check that B fits within the hardware limits of constant memory
    const int b_numel = B.numel();
    if (A.scalar_type() == torch::kFloat) {
        TORCH_CHECK(b_numel <= MAX_FLOAT_B_ELEMS, "B matrix too large to fit in constant memory for float.");
        cudaMemcpyToSymbol(B_const_float, B.data_ptr<float>(), b_numel * sizeof(float));
    } else if (A.scalar_type() == torch::kDouble) {
        TORCH_CHECK(b_numel <= MAX_DOUBLE_B_ELEMS, "B matrix too large to fit in constant memory for double.");
        cudaMemcpyToSymbol(B_const_double, B.data_ptr<double>(), b_numel * sizeof(double));
    }

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Set thread block and grid dimensions
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_const_kernel", ([&] {
        matmul_transpose_const_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_const_cuda, "Matrix multiplication with transposed inputs using constant memory (CUDA)");
}
