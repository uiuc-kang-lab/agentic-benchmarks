#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel performs matrix-vector multiplication using __ldg() for read-only accesses and vectorized loads for 128-bit alignment.
// Each block processes one matrix row and uses warp shuffle for efficient reduction.

template <typename scalar_t>
__global__ void ldg_aligned_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    int row = blockIdx.x;  // each block handles one row
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;  // warp id within block

    __shared__ scalar_t warp_results[32];
    scalar_t thread_sum = 0;

    // Obtain the raw pointers
    const scalar_t* A_ptr = A.data();
    const scalar_t* B_ptr = B.data();

    // Use vectorized loads for 128-bit aligned access if possible
    if constexpr (sizeof(scalar_t) == 4) {  // float case: use float4
        using vec_t = float4;
        int num_vec = K / 4;  // number of full 4-element groups
        
        // reinterpret the row pointer for A and the entire vector B
        const vec_t* A_vec = reinterpret_cast<const vec_t*>(A_ptr + row * K);
        const vec_t* B_vec = reinterpret_cast<const vec_t*>(B_ptr);

        // Loop over the vectorized portion
        for (int i = tid; i < num_vec; i += blockDim.x) {
            vec_t a_val = __ldg(&A_vec[i]);
            vec_t b_val = __ldg(&B_vec[i]);
            thread_sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
        }
        
        // Handle remaining elements
        int offset = num_vec * 4;
        for (int i = offset + tid; i < K; i += blockDim.x) {
            scalar_t a_val = __ldg(&A_ptr[row * K + i]);
            scalar_t b_val = __ldg(&B_ptr[i]);
            thread_sum += a_val * b_val;
        }
    } else if constexpr (sizeof(scalar_t) == 8) {  // double case: use double2
        using vec_t = double2;
        int num_vec = K / 2;  // number of full 2-element groups
        
        const vec_t* A_vec = reinterpret_cast<const vec_t*>(A_ptr + row * K);
        const vec_t* B_vec = reinterpret_cast<const vec_t*>(B_ptr);

        for (int i = tid; i < num_vec; i += blockDim.x) {
            vec_t a_val = __ldg(&A_vec[i]);
            vec_t b_val = __ldg(&B_vec[i]);
            thread_sum += a_val.x * b_val.x + a_val.y * b_val.y;
        }
        
        int offset = num_vec * 2;
        for (int i = offset + tid; i < K; i += blockDim.x) {
            scalar_t a_val = __ldg(&A_ptr[row * K + i]);
            scalar_t b_val = __ldg(&B_ptr[i]);
            thread_sum += a_val * b_val;
        }
    } else {
        // Fallback: element-wise load with __ldg()
        for (int i = tid; i < K; i += blockDim.x) {
            scalar_t a_val = __ldg(&A_ptr[row * K + i]);
            scalar_t b_val = __ldg(&B_ptr[i]);
            thread_sum += a_val * b_val;
        }
    }

    // Perform warp-level reduction using shuffle operations
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's lane 0 stores its partial result
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (wid == 0) {
        scalar_t final_sum = (lane < (blockDim.x >> 5)) ? warp_results[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
        if (lane == 0) {
            C[row][0] = final_sum;
        }
    }
}

// C++ interface function wrapping the CUDA kernel
torch::Tensor ldg_aligned_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch: B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;  // Use 256 threads per block
    // Launch one block per row
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ldg_aligned_matvec_mul_cuda", ([&] {
        ldg_aligned_matvec_kernel<scalar_t><<<M, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ldg_aligned_matvec_mul_cuda, "Matrix-Vector Multiplication with __ldg() and 128-bit Alignment (CUDA)");
}
