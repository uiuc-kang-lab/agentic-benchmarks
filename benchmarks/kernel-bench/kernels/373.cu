#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for matrix-vector multiplication combining vectorized loads and tiled reduction
// This kernel uses vectorized loads for memory efficiency and shared memory for block-level reduction

template <typename scalar_t>
__global__ void optimized_matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    int row = blockIdx.y;
    int tile_offset = blockIdx.x * blockDim.x;
    int tile_end = min(tile_offset + blockDim.x, K);

    scalar_t local_sum = 0;
    if (row < M) {
        if constexpr (sizeof(scalar_t) == 4) {  // float case: use float4
            using vec_t = float4;
            int num_vec = (tile_end - tile_offset) / 4;
            const vec_t* A_vec = reinterpret_cast<const vec_t*>(A.data() + row * K + tile_offset);
            const vec_t* B_vec = reinterpret_cast<const vec_t*>(B.data() + tile_offset);

            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                vec_t a_val = __ldg(&A_vec[i]);
                vec_t b_val = __ldg(&B_vec[i]);
                local_sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
            }

            int offset = num_vec * 4;
            for (int i = offset + tile_offset + threadIdx.x; i < tile_end; i += blockDim.x) {
                local_sum += A[row][i] * B[i];
            }
        } else if constexpr (sizeof(scalar_t) == 8) {  // double case: use double2
            using vec_t = double2;
            int num_vec = (tile_end - tile_offset) / 2;
            const vec_t* A_vec = reinterpret_cast<const vec_t*>(A.data() + row * K + tile_offset);
            const vec_t* B_vec = reinterpret_cast<const vec_t*>(B.data() + tile_offset);

            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                vec_t a_val = __ldg(&A_vec[i]);
                vec_t b_val = __ldg(&B_vec[i]);
                local_sum += a_val.x * b_val.x + a_val.y * b_val.y;
            }

            int offset = num_vec * 2;
            for (int i = offset + tile_offset + threadIdx.x; i < tile_end; i += blockDim.x) {
                local_sum += A[row][i] * B[i];
            }
        } else {
            for (int i = tile_offset + threadIdx.x; i < tile_end; i += blockDim.x) {
                local_sum += A[row][i] * B[i];
            }
        }
    }

    extern __shared__ scalar_t sdata[];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && row < M) {
        atomicAdd(&(C[row][0]), sdata[0]);
    }
}

// C++ interface function wrapping the optimized CUDA kernel
torch::Tensor optimized_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch: B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;
    int grid_x = (K + threads - 1) / threads;
    dim3 blocks(grid_x, M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_matvec_mul_cuda", ([&] {
        size_t sharedMemBytes = threads * sizeof(scalar_t);
        optimized_matvec_mul_kernel<scalar_t><<<blocks, threads, sharedMemBytes>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_matvec_mul_cuda, "Optimized Matrix-Vector Multiplication (CUDA)");
}
