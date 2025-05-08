#include <torch/extension.h>
#include <cuda_runtime.h>

// Combined kernel: uses shared memory for A and manual loop unrolling for dot product accumulation
__global__ void einsum_kernel_combined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Each block handles one (b, i, j) triple
    int index = blockIdx.x;
    int b = index / (I * J);
    int rem = index % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Allocate shared memory for A[b,i,j,*]
    extern __shared__ float shared_A[];  // size = L floats
    int tid = threadIdx.x;
    
    // Load the vector A[b,i,j,*] into shared memory in a coalesced manner
    for (int l = tid; l < L; l += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        shared_A[l] = A[a_index];
    }
    __syncthreads();

    // Each thread computes one or several C[b,i,j,k] by iterating over k with stride
    for (int k = tid; k < K; k += blockDim.x) {
        float sum = 0.0f;
        int l = 0;
        // Loop unrolling for improved performance over the L dimension
        #pragma unroll 4
        for (; l + 3 < L; l += 4) {
            int b_offset0 = l     * K + k;
            int b_offset1 = (l+1) * K + k;
            int b_offset2 = (l+2) * K + k;
            int b_offset3 = (l+3) * K + k;
            sum += shared_A[l]     * B[b_offset0] +
                   shared_A[l + 1] * B[b_offset1] +
                   shared_A[l + 2] * B[b_offset2] +
                   shared_A[l + 3] * B[b_offset3];
        }
        // Process remaining elements
        for (; l < L; ++l) {
            sum += shared_A[l] * B[l * K + k];
        }

        // Write the result to the output tensor
        int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
        C[c_index] = sum;
    }
}

// Forward function to launch the kernel

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "Tensor A must be 4D");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch: A.size(3) must equal B.size(0)");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // One block per (b, i, j) triple ensures that A[b,i,j,*] can be loaded into shared memory
    int numBlocks = BATCH * I * J;
    int threads = 256;
    size_t sharedMemBytes = L * sizeof(float);

    einsum_kernel_combined<<<numBlocks, threads, sharedMemBytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined shared memory load and loop unrolling kernel for einsum (CUDA)");
}
