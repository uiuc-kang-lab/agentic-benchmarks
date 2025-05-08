#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Shared memory for tile-based computation
    extern __shared__ float shared_mem[];
    float* shared_A = shared_mem;
    float* shared_B = &shared_mem[blockDim.x];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= BATCH * I * J * K) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;
    
    // Process input in tiles
    const int TILE_SIZE = 32;
    for (int tile = 0; tile < L; tile += TILE_SIZE) {
        // Collaborative loading of tiles into shared memory
        for (int t = threadIdx.x; t < TILE_SIZE && (tile + t) < L; t += blockDim.x) {
            // Load A tile
            if (b < BATCH && i < I && j < J) {
                shared_A[t] = A[b * I*J*L + i*J*L + j*L + (tile + t)];
            }
            // Load B tile
            if ((tile + t) < L) {
                shared_B[t] = B[(tile + t)*K + k];
            }
        }
        __syncthreads();

        // Compute using tiles
        for (int l = 0; l < TILE_SIZE && (tile + l) < L; ++l) {
            sum += shared_A[l] * shared_B[l];
        }
        __syncthreads();
    }

    if (global_idx < BATCH * I * J * K) {
        C[global_idx] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total_elements = BATCH * I * J * K;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}