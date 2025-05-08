#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define VECTOR_SIZE 4

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_scan(scalar_t val) {
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t tmp = __shfl_down_sync(0xffffffff, val, offset);
        if (lane >= offset) val += tmp;
    }
    return val;
}

template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    constexpr int elements_per_thread = VECTOR_SIZE;
    const int elements_total = (L + elements_per_thread - 1) / elements_per_thread;
    const int stride = blockDim.x * elements_per_thread;

    scalar_t local_sum = 0;
    for (int base = 0; base < elements_total; base += blockDim.x) {
        int idx = base * elements_per_thread + tid * elements_per_thread;
        
        // Vectorized load
        scalar_t vals[elements_per_thread];
        bool masks[elements_per_thread];
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            if (idx + i < L) {
                vals[i] = x_row[idx + i];
                masks[i] = mask_row[idx + i];
            }
        }

        // Thread-local cumulative sum
        scalar_t thread_sum = 0;
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            if (idx + i < L) {
                thread_sum += masks[i] ? vals[i] : 0;
                vals[i] = thread_sum;
            }
        }

        // Warp-level scan
        scalar_t warp_total = warp_scan(thread_sum);
        scalar_t warp_offset = warp_total - thread_sum;

        // Store results with vectorized writes
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            if (idx + i < L) {
                vals[i] += (local_sum + warp_offset);
                output_row[idx + i] = vals[i];
            }
        }

        local_sum += __shfl_sync(0xffffffff, warp_total, 0);
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

    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    auto transposed = x.transpose(dim, -1).contiguous();
    auto mask_transposed = mask.transpose(dim, -1).contiguous();

    int64_t N = transposed.numel() / transposed.size(-1);
    int64_t L = transposed.size(-1);

    auto output_flat = torch::empty_like(transposed);

    dim3 blocks(N);
    dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            transposed.data_ptr<scalar_t>(),
            mask_transposed.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    return output_flat.transpose(dim, -1).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Warp-optimized Masked Cumulative Sum (CUDA)");
}