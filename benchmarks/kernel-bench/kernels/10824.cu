#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Define number of rows processed per block in the y-dimension
#define ROWS_PER_BLOCK 8

// 2D Indexed Masked Cumulative Sum Kernel
// Each block processes ROWS_PER_BLOCK rows concurrently. The x-dimension (tile) covers a contiguous segment of columns.
// For each row, the kernel iterates over tiles of columns. Within each tile, the Hillis-Steele scan is performed in shared memory,
// and a running offset is applied to produce the full cumulative sum. This mapping uses 2D thread indexing to reduce kernel launch overhead
// and maximizes occupancy by processing multiple rows per block, ensuring an efficient mapping of both rows and columns.

template <typename scalar_t>
__global__ void block2d_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    // Each block processes ROWS_PER_BLOCK rows concurrently
    int row = blockIdx.x * blockDim.y + threadIdx.y; // Global row index
    int tile_width = blockDim.x; // Number of threads in x-dimension defines the tile width

    extern __shared__ char shared_mem[];
    // Shared memory organized as [ROWS_PER_BLOCK][tile_width]
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);

    if (row >= N) return;

    int num_tiles = (L + tile_width - 1) / tile_width;
    scalar_t running_sum = static_cast<scalar_t>(0);

    // Process one tile (a contiguous block of columns) at a time
    for (int t = 0; t < num_tiles; t++) {
        int col_start = t * tile_width;
        // Compute count: number of valid columns in this tile
        int count = (col_start + tile_width <= L) ? tile_width : (L - col_start);
        int col = col_start + threadIdx.x;

        // Each thread loads one element if within the valid range, else loads 0
        scalar_t val = static_cast<scalar_t>(0);
        if (threadIdx.x < count && col < L) {
            val = mask[row * L + col] ? x[row * L + col] : static_cast<scalar_t>(0);
        }
        // Store into shared memory; each row occupies a contiguous segment of size 'tile_width'
        sdata[threadIdx.y * tile_width + threadIdx.x] = val;
        __syncthreads();

        // In-tile inclusive scan using the Hillis-Steele algorithm
        for (int offset = 1; offset < tile_width; offset *= 2) {
            scalar_t temp = static_cast<scalar_t>(0);
            if (threadIdx.x >= offset && threadIdx.x < count) {
                temp = sdata[threadIdx.y * tile_width + threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x < count) {
                sdata[threadIdx.y * tile_width + threadIdx.x] += temp;
            }
            __syncthreads();
        }

        // Write out the scanned values for valid columns, adding the running offset
        if (threadIdx.x < count) {
            output[row * L + col] = sdata[threadIdx.y * tile_width + threadIdx.x] + running_sum;
        }

        // Update the running offset for this row with the last element of this tile
        if (threadIdx.x == count - 1) {
            running_sum += sdata[threadIdx.y * tile_width + threadIdx.x];
        }
        __syncthreads();
    }
}


// Macros to check inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Host function to launch the 2D indexed masked cumulative sum kernel
// It permutes the input tensor such that the dimension to be scanned is the last dimension,
// reshapes the tensor to [N, L], and launches the kernel with a 2D block configuration.

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust dimension if negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dimension to the last axis
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape to a 2D tensor with shape [N, L]
    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Configure block and grid dimensions
    // We'll use a 2D block: blockDim.x = tile_width (set to 256) and blockDim.y = ROWS_PER_BLOCK
    int tile_width = 256;
    dim3 block(tile_width, ROWS_PER_BLOCK);
    int grid_x = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(grid_x);

    // Calculate shared memory size: one tile per row in the block
    size_t shared_size = tile_width * ROWS_PER_BLOCK * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "block2d_masked_cumsum_cuda", ([&] {
        shared_size = tile_width * ROWS_PER_BLOCK * sizeof(scalar_t);
        block2d_masked_cumsum_kernel<scalar_t><<<grid, block, shared_size>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    // Reshape and permute the output back to the original layout
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "2D Indexed Masked Cumulative Sum (CUDA)");
}
