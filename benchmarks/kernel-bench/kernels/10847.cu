#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ALIGN_MASK 0xFFFFFFFFFFFFFFF0  // 16-byte alignment mask

template <typename scalar_t>
__global__ void aligned_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ char shared_mem[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(shared_mem);
    bool* s_mask = reinterpret_cast<bool*>(reinterpret_cast<char*>(s_data) + (((L * sizeof(scalar_t) + 15) & ALIGN_MASK)));

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= N) return;

    // Align pointers for coalesced memory access
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    // Load data into shared memory using __ldg for read-only data
    // Process 4 elements per iteration when possible for better memory coalescing
    #pragma unroll 4
    for (int i = tid; i < L; i += BLOCK_SIZE) {
        s_data[i] = __ldg(x_row + i);
        s_mask[i] = __ldg(mask_row + i);
    }
    __syncthreads();

    // Single thread per row computes the cumulative sum
    // This avoids warp divergence and maintains sequential addition for accuracy
    if (tid == 0) {
        scalar_t sum = 0;
        
        // Process aligned chunks of 4 elements when possible
        int aligned_length = L & ~3;  // Round down to multiple of 4
        int i = 0;
        
        // Process 4 elements at a time
        for (; i < aligned_length; i += 4) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (s_mask[i + j]) {
                    sum += s_data[i + j];
                }
                output_row[i + j] = sum;
            }
        }
        
        // Handle remaining elements
        for (; i < L; i++) {
            if (s_mask[i]) {
                sum += s_data[i];
            }
            output_row[i] = sum;
        }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    const int threads = BLOCK_SIZE;
    const int blocks = N;
    
    // Ensure shared memory is properly aligned
    const size_t aligned_scalar_size = (L * sizeof(scalar_t) + 15) & ALIGN_MASK;
    const size_t aligned_bool_size = (L * sizeof(bool) + 15) & ALIGN_MASK;
    const size_t shared_mem_size = aligned_scalar_size + aligned_bool_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "aligned_masked_cumsum_cuda", ([&] {
        aligned_masked_cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    auto output = output_permuted.permute(inv_perm);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Aligned Masked Cumulative Sum (CUDA)");
}