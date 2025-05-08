#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 1024
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void streamed_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int64_t chunk_start,
    const int64_t chunk_size,
    const int64_t L) {

    int row = blockIdx.x;
    int global_row = chunk_start + row;
    if(row >= chunk_size) return;

    const scalar_t* x_row = x + global_row * L;
    const bool* mask_row = mask + global_row * L;
    scalar_t* out_row = output + global_row * L;

    if(L <= blockDim.x) {
        int tid = threadIdx.x;
        int lane = tid & (WARP_SIZE - 1);
        int warpId = tid >> 5;

        __shared__ scalar_t warp_results[32];
        
        scalar_t val = 0;
        if(tid < L) {
            val = mask_row[tid] ? x_row[tid] : 0;
        }

        #pragma unroll
        for(int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            scalar_t n = __shfl_up_sync(0xffffffff, val, offset);
            if(lane >= offset) val += n;
        }

        if(lane == WARP_SIZE-1 || tid == L-1) {
            warp_results[warpId] = val;
        }
        __syncthreads();

        if(warpId > 0) {
            scalar_t prefix = 0;
            if(lane == 0) {
                for(int i = 0; i < warpId; i++) {
                    prefix += warp_results[i];
                }
            }
            prefix = __shfl_sync(0xffffffff, prefix, 0);
            val += prefix;
        }

        if(tid < L) {
            out_row[tid] = val;
        }
    } else {
        if(threadIdx.x == 0) {
            scalar_t sum = 0;
            for(int64_t i = 0; i < L; i++) {
                if(mask_row[i]) sum += x_row[i];
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

    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threads = (L <= 256) ? std::min(static_cast<int64_t>(256), L) : 256;
    threads = (threads + WARP_SIZE - 1) & ~(WARP_SIZE - 1);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        for(int64_t chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
            int64_t chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
            int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
            
            dim3 grid(chunk_size);
            dim3 block(threads);

            streamed_masked_cumsum_kernel<scalar_t><<<grid, block, 0, streams[stream_idx]>>>(
                x_flat.data_ptr<scalar_t>(),
                mask_flat.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>(),
                chunk_start,
                chunk_size,
                L
            );
        }
    }));

    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Streamed Masked Cumulative Sum (CUDA)");
}