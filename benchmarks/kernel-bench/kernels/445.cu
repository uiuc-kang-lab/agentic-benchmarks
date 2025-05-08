#include <torch/extension.h>
#include <cooperative_groups.h>

constexpr int VEC_WIDTH = 4;
constexpr int BLOCK_SIZE = 128;

namespace cg = cooperative_groups;

template <typename scalar_t, int vec_width>
struct VectorizedAccess {
    using VecType = typename std::conditional<std::is_same<scalar_t, float>::value, float4, double2>::type;
    __device__ __forceinline__ static VecType load(const scalar_t* ptr) {
        return *reinterpret_cast<const VecType*>(ptr);
    }
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce(scalar_t val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Removed the block_reduce function that used unsupported cooperative_groups features

template <typename scalar_t, int vec_width>
__global__ void matvec_mul_kernel(const scalar_t* A, const scalar_t* B, scalar_t* C, int64_t M, int64_t K) {
    constexpr int vec_elements = sizeof(typename VectorizedAccess<scalar_t, vec_width>::VecType) / sizeof(scalar_t);
    __shared__ scalar_t smem[BLOCK_SIZE];
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int64_t row = blockIdx.x;
    const int64_t tid = block.thread_rank();
        
    if (row >= M) return;
    
    const scalar_t* row_ptr = A + row * K;
    scalar_t thread_sum = 0;
    
    for (int64_t base = 0; base < K; base += BLOCK_SIZE * vec_elements) {
        int64_t k = base + tid * vec_elements;
        if (k + vec_elements <= K) {
            auto a_vec = VectorizedAccess<scalar_t, vec_width>::load(row_ptr + k);
            auto b_vec = VectorizedAccess<scalar_t, vec_width>::load(B + k);
            
#pragma unroll
            for (int i = 0; i < vec_elements; ++i)
                thread_sum += reinterpret_cast<scalar_t*>(&a_vec)[i] * reinterpret_cast<scalar_t*>(&b_vec)[i];
        } else {
            for (int i = 0; k + i < K && i < vec_elements; ++i)
                thread_sum += row_ptr[k + i] * B[k + i];
        }
    }
    
    thread_sum = warp_reduce(thread_sum);
    
    if (warp.thread_rank() == 0)
        smem[warp.meta_group_rank()] = thread_sum;
    
    __syncthreads();
    
    if (tid < 32) {
        scalar_t block_sum = (tid < block.size() / 32) ? smem[tid] : 0;
        block_sum = warp_reduce(block_sum);
        
        if (tid == 0)
            C[row] = block_sum;
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    
    auto M = A.size(0);
    auto K = A.size(1);
    auto C = torch::zeros({M}, A.options());
    
    dim3 blocks(M);
    dim3 threads(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", [&] {
        matvec_mul_kernel<scalar_t, VEC_WIDTH><<<blocks, threads>>>(
            A.contiguous().data_ptr<scalar_t>(),
            B.contiguous().view(-1).data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    });
    
    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}
