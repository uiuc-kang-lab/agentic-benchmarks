#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__forceinline__ __device__ scalar_t ldg(const scalar_t* ptr) {
    return __ldg(ptr);
}

template <typename scalar_t>
__global__ void aligned_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int64_t N,
    const int64_t L) {

    const int row = blockIdx.x;
    if (row >= N) return;

    // Align pointers for the current row
    const scalar_t* __restrict__ x_row = x + row * L;
    const bool* __restrict__ mask_row = mask + row * L;
    scalar_t* __restrict__ out_row = output + row * L;

    if (L <= 256) {  // Use parallel warp-scan for small rows
        const int tid = threadIdx.x;
        const int lane = tid & 31;
        const int warpId = tid >> 5;

        // Load data using __ldg for read-only access
        scalar_t val = 0;
        if (tid < L) {
            val = ldg(x_row + tid) * static_cast<scalar_t>(ldg(mask_row + tid));
        }

        // Warp-level scan using shuffle
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            scalar_t n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val += n;
        }

        // Store warp results in shared memory
        __shared__ scalar_t warpSums[8];  // Support up to 8 warps (256 threads)
        if (lane == 31 || tid == L - 1) {
            warpSums[warpId] = val;
        }
        __syncthreads();

        // Add previous warp sums
        if (warpId > 0) {
            scalar_t warpOffset = 0;
            if (lane == 0) {
                #pragma unroll
                for (int i = 0; i < warpId; i++) {
                    warpOffset += warpSums[i];
                }
            }
            warpOffset = __shfl_sync(0xffffffff, warpOffset, 0);
            val += warpOffset;
        }

        // Write result
        if (tid < L) {
            out_row[tid] = val;
        }
    } else {
        // For longer rows, use vectorized loads where possible
        if (threadIdx.x == 0) {
            scalar_t sum = 0;
            
            // Process aligned chunks using float4
            const int vec_size = 4;
            const int aligned_length = (L / vec_size) * vec_size;
            
            for (int i = 0; i < aligned_length; i += vec_size) {
                float4 x_vec = *reinterpret_cast<const float4*>(x_row + i);
                bool4 mask_vec = *reinterpret_cast<const bool4*>(mask_row + i);
                
                // Process each element in the vector
                if (mask_vec.x) sum += x_vec.x;
                out_row[i] = sum;
                
                if (mask_vec.y) sum += x_vec.y;
                out_row[i + 1] = sum;
                
                if (mask_vec.z) sum += x_vec.z;
                out_row[i + 2] = sum;
                
                if (mask_vec.w) sum += x_vec.w;
                out_row[i + 3] = sum;
            }
            
            // Handle remaining elements
            for (int i = aligned_length; i < L; ++i) {
                if (ldg(mask_row + i)) {
                    sum += ldg(x_row + i);
                }
                out_row[i] = sum;
            }
        }
    }
}

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Configure kernel launch
    const int threads = (L <= 256) ? 256 : 1;
    const int blocks = N;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        aligned_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    return output_permuted.permute(inv_perm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Aligned Masked Cumulative Sum with LDG (CUDA)");
}