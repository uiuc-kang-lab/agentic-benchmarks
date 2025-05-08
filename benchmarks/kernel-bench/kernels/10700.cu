#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hybrid_reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer,
    int64_t n,
    int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int dim) {

    // For last dimension, use warp-level optimizations
    if (dim == ndim - 1 && n <= 1024) {
        int row = blockIdx.x;
        const int64_t row_offset = row * n;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp_id = tid >> 5;

        // Load using __ldg for better cache utilization
        scalar_t val = 0;
        if (tid < n) {
            val = __ldg(&input[row_offset + (n - 1 - tid)]);
        }

        // Warp-level scan using shuffle
        for (int offset = 1; offset < 32; offset *= 2) {
            scalar_t tmp = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) {
                val += tmp;
            }
        }

        // Inter-warp communication using shared memory
        __shared__ scalar_t warp_sums[32];
        if (tid < n && (lane == 31 || tid == n - 1)) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();

        if (tid < n) {
            scalar_t warp_offset = 0;
            for (int w = 0; w < warp_id; w++) {
                warp_offset += warp_sums[w];
            }
            output[row_offset + (n - 1 - tid)] = val + warp_offset;
        }
    }
    // For other dimensions, use grid-stride loop approach
    else {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t gridStride = blockDim.x * gridDim.x;
        
        for (int64_t r = idx; r < outer; r += gridStride) {
            int64_t offset = 0;
            int64_t tmp = r;
            for (int d = ndim - 1; d >= 0; d--) {
                if (d == dim) continue;
                int64_t cur_size = sizes[d];
                int64_t idx_d = tmp % cur_size;
                tmp /= cur_size;
                offset += idx_d * strides[d];
            }

            int64_t stride_dim = strides[dim];
            scalar_t cum = 0;
            for (int64_t j = n - 1; j >= 0; j--) {
                int64_t cur_index = offset + j * stride_dim;
                cum += __ldg(&input[cur_index]);
                output[cur_index] = cum;
            }
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::empty_like(x);
    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    // Prepare dimension information
    int64_t h_sizes[16], h_strides[16];
    for (int i = 0; i < ndim; i++) {
        h_sizes[i] = x.size(i);
        h_strides[i] = x.stride(i);
    }

    int64_t *d_sizes, *d_strides;
    cudaMalloc(&d_sizes, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_sizes, h_sizes, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, h_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int threads = (dim == ndim - 1 && n <= 1024) ? 
                 std::min(1024, (int)next_power_of_2(n)) : 256;
    int blocks = (dim == ndim - 1) ? outer : (outer + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hybrid_reverse_cumsum", ([&] {
        hybrid_reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            n,
            ndim,
            d_sizes,
            d_strides,
            dim);
    }));

    cudaFree(d_sizes);
    cudaFree(d_strides);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Hybrid reverse cumulative sum (CUDA)");
}