#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int BLOCK_SIZE = 256, int ITEMS_PER_THREAD = 4>
__global__ void shared_mem_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    
    if (row >= N) return;
    
    // Point to current row data
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;
    
    // Initialize running sum for this row
    scalar_t running_sum = 0;
    
    // Process the row in tiles
    for (int base = 0; base < L; base += BLOCK_SIZE) {
        shared_data[tid] = 0;
        
        // Load and process data in current tile
        if (base + tid < L) {
            if (mask_row[base + tid]) {
                shared_data[tid] = x_row[base + tid];
            }
        }
        
        __syncthreads();
        
        // Perform parallel prefix sum within the block
        for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
            scalar_t temp = 0;
            if (tid >= stride) {
                temp = shared_data[tid - stride];
            }
            __syncthreads();
            
            if (tid >= stride) {
                shared_data[tid] += temp;
            }
            __syncthreads();
        }
        
        // Write results back to global memory with offset from previous tiles
        if (base + tid < L) {
            output_row[base + tid] = shared_data[tid] + running_sum;
        }
        
        // Update running sum for next tile
        __syncthreads();
        if (base + BLOCK_SIZE <= L) {
            running_sum += shared_data[BLOCK_SIZE - 1];
        } else if (L - base > 0) {
            running_sum += shared_data[(L - base) - 1];
        }
    }
}

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_CUDA(x);
    CHECK_CUDA(mask);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    constexpr int BLOCK_SIZE = 256;
    const int blocks = N;  // One block per row

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        shared_mem_cumsum_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum (CUDA)");
}