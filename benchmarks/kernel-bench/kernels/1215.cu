#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that computes C = einsum('bijl,lk->bijk')
// using one block per (b, i, j) and shared memory for A[b,i,j,:]

__global__ void einsum_kernel_coalesced_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Compute indices for b, i, j based on blockIdx
    int index = blockIdx.x;
    int b = index / (I * J);
    int rem = index % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Allocate shared memory for the entire vector A[b,i,j,*]
    extern __shared__ float shared_A[];  // size = L floats

    // Load A[b,i,j,*] into shared memory in a coalesced manner
    int tid = threadIdx.x;
    for (int l = tid; l < L; l += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        shared_A[l] = A[a_index];
    }
    __syncthreads();

    // Each thread computes one or more elements in the K dimension
    // Ensure coalesced access to B and C: threads in a warp use consecutive k indices.
    for (int k = tid; k < K; k += blockDim.x) {
        float sum = 0.0f;
        for (int l = 0; l < L; ++l) {
            // B is [L x K]: access element (l, k) = B[l*K + k]
            sum += shared_A[l] * B[l * K + k];
        }
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

    // Each block processes one (b, i, j) triple
    int numBlocks = BATCH * I * J;
    int threads = 256;

    // Shared memory to hold A[b,i,j,*]
    size_t sharedMemBytes = L * sizeof(float);

    einsum_kernel_coalesced_shared<<<numBlocks, threads, sharedMemBytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with coalesced memory accesses using shared memory (CUDA)");
}
