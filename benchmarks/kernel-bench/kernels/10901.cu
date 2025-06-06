#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct Float4 { float x, y, z, w; };
struct Bool4 { bool x, y, z, w; };

template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t start_row,
    int64_t num_rows,
    int64_t L) {
    
    // Use thread coalescing for better memory access patterns
    const int64_t tid = threadIdx.x;
    const int64_t bid = blockIdx.x;
    const int64_t row = bid * blockDim.x + tid + start_row;
    
    if (row >= start_row + num_rows) return;

    // Adjust base pointers for coalesced access
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    scalar_t sum = 0;
    const int64_t vec_steps = L / 4;
    
    // Process 4 elements at a time
    for (int64_t i = 0; i < vec_steps; ++i) {
        Float4 vals = *reinterpret_cast<const Float4*>(x_row + i*4);
        Bool4 masks = *reinterpret_cast<const Bool4*>(mask_row + i*4);

        if (masks.x) sum += vals.x;
        output_row[i*4] = sum;
        if (masks.y) sum += vals.y;
        output_row[i*4+1] = sum;
        if (masks.z) sum += vals.z;
        output_row[i*4+2] = sum;
        if (masks.w) sum += vals.w;
        output_row[i*4+3] = sum;
    }

    // Process remaining elements
    for (int64_t i = vec_steps*4; i < L; ++i) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        output_row[i] = sum;
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

    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    auto x_permuted = x.transpose(dim, -1).contiguous();
    auto mask_permuted = mask.transpose(dim, -1).contiguous();

    const int64_t N = x_permuted.numel() / x_permuted.size(-1);
    const int64_t L = x_permuted.size(-1);
    auto output_flat = torch::empty_like(x_permuted);

    constexpr int num_streams = 4;
    constexpr int threads_per_block = 128;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int64_t chunk_size = (N + num_streams - 1) / num_streams;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum", [&] {
        for (int s = 0; s < num_streams; ++s) {
            const int64_t start = s * chunk_size;
            const int64_t end = std::min(start + chunk_size, N);
            const int64_t n_rows = end - start;
            if (n_rows <= 0) continue;

            const int blocks = (n_rows + threads_per_block - 1) / threads_per_block;
            masked_cumsum_kernel<scalar_t><<<blocks, threads_per_block, 0, streams[s]>>>(
                x_permuted.data_ptr<scalar_t>(),
                mask_permuted.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>(),
                start,
                n_rows,
                L
            );
        }
    });

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return output_flat.transpose(dim, -1).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Vectorized masked cumsum with stream parallelism");
}
