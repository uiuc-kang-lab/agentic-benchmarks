#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with shared memory optimization for matrix-vector multiplication
// Each block loads a portion of the vector B into shared memory to reduce global memory accesses

template <typename scalar_t>
__global__ void matvec_mul_kernel_shared(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    extern __shared__ char smem[];
    scalar_t* shared_B = reinterpret_cast<scalar_t*>(smem);  // Shared memory for vector B

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Load vector B into shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        shared_B[i] = B[i];
    }
    __syncthreads();

    if (row < M) {
        scalar_t sum = 0;
        for (int k = lane; k < K; k += 32) {
            sum += A[row][k] * shared_B[k];
        }

        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane == 0) {
            C[row][0] = sum;
        }
    }
}

// C++ function that wraps the CUDA kernel

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);

    // Check dimensions
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K, 1)");

    // Flatten B to a 1D tensor
    auto B_flat = B.view({-1});

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Set block and grid sizes
    int threads = 128;
    int blocks = (M + threads - 1) / threads;

    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_shared<scalar_t><<<blocks, threads, K * sizeof(scalar_t)>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    // Ensure synchronization
    cudaDeviceSynchronize();

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication with Shared Memory Optimization (CUDA)");
}
