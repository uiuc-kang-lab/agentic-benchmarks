#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CONST_MAX 1024

// Declare constant memory for vector B (read-only data)
__constant__ float d_const_B_float[CONST_MAX];
__constant__ double d_const_B_double[CONST_MAX];

// Helper function to load B from constant memory
template <typename scalar_t>
__device__ inline scalar_t get_const_B(int index);

template <>
__device__ inline float get_const_B<float>(int index) {
    return d_const_B_float[index];
}

template <>
__device__ inline double get_const_B<double>(int index) {
    return d_const_B_double[index];
}

// CUDA kernel for matrix-vector multiplication using constant memory for B
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    __shared__ scalar_t shared_sum[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row < M) {
        scalar_t thread_sum = 0;
        // Each thread processes a slice of the K dimension
        for (int64_t k = tid; k < K; k += blockDim.x) {
            thread_sum += A[row][k] * get_const_B<scalar_t>(k);
        }
        shared_sum[tid] = thread_sum;
        __syncthreads();

        // Parallel reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[row][0] = shared_sum[0];
        }
    }
}

// C++ interface function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
                "B must be a vector of shape (K,) or (K, 1)");

    // Flatten B to be 1-D
    auto B_flat = B.view({-1});

    // Ensure B fits within our constant memory limit
    TORCH_CHECK(K <= CONST_MAX, "Size of vector B exceeds constant memory limit");

    // Copy B into constant memory based on its type
    AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "copy_B_to_constant", ([&] {
        if (std::is_same<scalar_t, float>::value) {
            cudaMemcpyToSymbol(d_const_B_float, B_flat.data_ptr<scalar_t>(), K * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyToSymbol(d_const_B_double, B_flat.data_ptr<scalar_t>(), K * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
        }
    }));

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Define block and grid sizes: one block per row
    int threads = 256;
    dim3 blocks(M);

    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication with constant memory for B (CUDA)");
}
