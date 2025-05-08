#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel using grid-stride loop to ensure uniform control flow across threads and unrolls the inner loop
__global__ void einsum_kernel_grid_stride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Total number of output elements
    int total = BATCH * I * J * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Precompute strides to reduce redundant arithmetic
    const int stride_A = I * J * L;      // Stride to next batch in A
    const int stride_A_row = J * L;        // Stride to next row (i) in A
    const int stride_B = K;                // Stride to next element in a row of B

    // Each thread processes a uniform set of elements via a grid-stride loop
    for(int index = idx; index < total; index += stride) {
        // Decompose index into 4D coordinates
        int k = index % K;
        int tmp = index / K;
        int j = tmp % J;
        tmp /= J;
        int i = tmp % I;
        int b = tmp / I;
        
        float sum = 0.0f;
        int baseA = b * stride_A + i * stride_A_row + j * L;

        // Unroll inner loop by factor of 4 to reduce loop overhead
        int l = 0;
        int L4 = (L / 4) * 4;  // largest multiple of 4 less than or equal to L
        for(; l < L4; l += 4) {
            sum += A[baseA + l]     * B[l * stride_B + k] +
                   A[baseA + l + 1] * B[(l + 1) * stride_B + k] +
                   A[baseA + l + 2] * B[(l + 2) * stride_B + k] +
                   A[baseA + l + 3] * B[(l + 3) * stride_B + k];
        }
        for(; l < L; l++) {
            sum += A[baseA + l] * B[l * stride_B + k];
        }

        C[index] = sum;
    }
}

// Host function to launch the kernel

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

    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total = BATCH * I * J * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    einsum_kernel_grid_stride<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Uniform grid-stride 4D tensor-matrix multiplication (CUDA)");
}
