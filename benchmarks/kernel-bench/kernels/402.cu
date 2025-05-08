#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with warp-level cooperation for matrix-vector multiplication
// Each warp is responsible for computing one output element (one row dot product) 
// with threads in the warp loading consecutive elements from the row for coalesced access.

// Shared memory is utilized for partial results, and synchronization is minimized.

template <typename scalar_t>
__global__ void matvec_mul_kernel_shared(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    // Declare shared memory for this block
    extern __shared__ scalar_t shared[];
    scalar_t* shared_B = shared;
    int warp_id = threadIdx.x / 32;           // warp index within the block
    int lane = threadIdx.x % 32;                // lane index within the warp
    int warpsPerBlock = blockDim.x / 32;         
    int row = blockIdx.x * warpsPerBlock + warp_id;  // global row index

    if (lane < K && row < M) {
        // Load B into shared memory
        shared_B[lane] = B[lane];
    }

    __syncthreads();  // Ensure shared memory is populated

    if (row < M) {
        scalar_t sum = static_cast<scalar_t>(0);
        // Loop over the row, letting each thread in the warp handle a subset of columns
        for (int col = lane; col < K; col += 32) {
            sum += A[row][col] * shared_B[col];
        }

        // Warp-level reduction using shuffle intrinsic
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // The first thread in the warp writes the result
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
    // Use 128 threads per block (i.e., 4 warps per block)
    int threads = 128;
    int warpsPerBlock = threads / 32;
    int blocks = (M + warpsPerBlock - 1) / warpsPerBlock;

    size_t shared_memory_bytes = K * sizeof(typename std::remove_pointer<decltype(A.data())>::type);

    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_shared<scalar_t><<<blocks, threads, shared_memory_bytes>>>(
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
