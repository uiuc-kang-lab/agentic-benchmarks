#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel combines vectorized __ldg() loads with tiling and warp-level reduction.
// It splits each row of A into tiles along the K-dimension so that multiple blocks can work in parallel,
// and each block uses vectorized loads when possible (float4 for floats, double2 for doubles).
// Partial sums are reduced within warps using __shfl_down_sync and then across warps using shared memory,
// and the final result for the tile is atomically added to the output C. This design leverages both the
// memory efficiency of read-only caching and the parallel reduction benefits of tiling.


// CUDA kernel: each block processes a tile of columns for a single row of A

template <typename scalar_t>
__global__ void hybrid_ldg_tile_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    // Each block processes a tile of the columns for a given row.
    int row = blockIdx.y;  // row index
    int tile_offset = blockIdx.x * blockDim.x;  // starting column index for this tile
    int tile_end = tile_offset + blockDim.x;
    if (tile_end > K) tile_end = K;

    scalar_t local_sum = 0;
    const scalar_t* A_ptr = A.data();
    const scalar_t* B_ptr = B.data();

    // Use vectorized loads when possible for float and double types
    if constexpr (sizeof(scalar_t) == 4) {  // float case: use float4
        using vec_t = float4;
        // Compute the number of full vectorized loads in the tile
        int num_vec = (tile_end - tile_offset) / 4;
        int offset_vec = tile_offset / 4;  // tile_offset is assumed to be divisible by 4

        // Reinterpret the pointers for A and B for vectorized loads
        const vec_t* A_vec = reinterpret_cast<const vec_t*>(A_ptr + row * K);
        const vec_t* B_vec = reinterpret_cast<const vec_t*>(B_ptr);

        // Loop over the vectorized portion of the tile
        for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
            vec_t a_val = __ldg(&A_vec[offset_vec + i]);
            vec_t b_val = __ldg(&B_vec[offset_vec + i]);
            local_sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
        }

        // Process any remaining elements in the tile
        int remainder_start = tile_offset + num_vec * 4;
        for (int j = remainder_start + threadIdx.x; j < tile_end; j += blockDim.x) {
            local_sum += __ldg(&A_ptr[row * K + j]) * __ldg(&B_ptr[j]);
        }

    } else if constexpr (sizeof(scalar_t) == 8) {  // double case: use double2
        using vec_t = double2;
        int num_vec = (tile_end - tile_offset) / 2;
        int offset_vec = tile_offset / 2;
        const vec_t* A_vec = reinterpret_cast<const vec_t*>(A_ptr + row * K);
        const vec_t* B_vec = reinterpret_cast<const vec_t*>(B_ptr);

        for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
            vec_t a_val = __ldg(&A_vec[offset_vec + i]);
            vec_t b_val = __ldg(&B_vec[offset_vec + i]);
            local_sum += a_val.x * b_val.x + a_val.y * b_val.y;
        }

        int remainder_start = tile_offset + num_vec * 2;
        for (int j = remainder_start + threadIdx.x; j < tile_end; j += blockDim.x) {
            local_sum += __ldg(&A_ptr[row * K + j]) * __ldg(&B_ptr[j]);
        }
    } else {
        // Fallback for other types: element-wise multiplication
        for (int j = tile_offset + threadIdx.x; j < tile_end; j += blockDim.x) {
            local_sum += __ldg(&A_ptr[row * K + j]) * __ldg(&B_ptr[j]);
        }
    }

    // Reduce local_sum within each warp using warp shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's leader stores its sum into shared memory
    extern __shared__ char shared[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);
    int lane = threadIdx.x & 31;  // warp lane
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction of warp sums across the block
    int numWarps = blockDim.x / 32;
    if (threadIdx.x < numWarps) {
        scalar_t warp_sum = sdata[threadIdx.x];
        // Reduce within the subset of warp leaders
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }
        if (threadIdx.x == 0) {
            // Atomically accumulate the block's partial sum into the final result
            atomicAdd(&C[row][0], warp_sum);
        }
    }
}


// Host function wrapping the hybrid kernel

torch::Tensor hybrid_ldg_tile_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch: B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Each block processes a tile of columns (tile size = blockDim.x) for a given row.
    // Grid dimensions: grid.x covers tiles along K, grid.y covers the M rows.
    int threads = 256;
    int grid_x = (K + threads - 1) / threads;
    dim3 blocks(grid_x, M);
    int warpsPerBlock = (threads + 31) / 32;
    size_t sharedMemBytes = warpsPerBlock * sizeof(float);  // updated in dispatch below

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_ldg_tile_cuda", ([&] {
        sharedMemBytes = warpsPerBlock * sizeof(scalar_t);
        hybrid_ldg_tile_kernel<scalar_t><<<blocks, threads, sharedMemBytes>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));

    return C;
}

// PyBind11 binding code

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_ldg_tile_cuda, "Hybrid Matrix-Vector Multiplication with Vectorized Loads and Warp-Level Reduction (CUDA)");
}
