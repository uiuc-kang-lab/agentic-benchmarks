#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void warp_optimized_reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer,
    int64_t n,
    int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int dim) {

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    
    // Compute base offset for this row
    int64_t offset = 0;
    if (dim == ndim - 1) {
        offset = row * n;
    } else {
        int64_t tmp = row;
        for (int d = ndim - 1; d >= 0; d--) {
            if (d == dim) continue;
            int64_t cur_size = sizes[d];
            int64_t idx_d = tmp % cur_size;
            tmp /= cur_size;
            offset += idx_d * strides[d];
        }
    }
    
    int64_t stride_dim = (dim == ndim - 1) ? 1 : strides[dim];

    // Fast path for small n using warp-level operations
    if (n <= 1024) {
        scalar_t val = 0;
        if (tid < n) {
            val = __ldg(&input[offset + (n - 1 - tid) * stride_dim]);
        }

        // Warp-level inclusive scan
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
            output[offset + (n - 1 - tid) * stride_dim] = val + warp_offset;
        }
    }
    // Fallback path for larger n using grid-stride approach
    else {
        int64_t gridStride = blockDim.x * gridDrid.x;
        for (int64_t j = n - 1; j >= 0; j--) {
            int64_t cur_index = offset + j * stride_dim;
            
            // Use atomicAdd for thread safety in the grid-stride case
            scalar_t val = __ldg(&input[cur_index]);
            atomicAdd(&output[cur_index], val);
        }
    }
}

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    auto output = at::zeros_like(x);  // Initialize with zeros for atomic adds

    int64_t n = x.size(dim);
    int64_t outer = x.numel() / n;

    // Launch parameters
    const int threads = (n <= 1024) ? next_power_of_2(n) : 256;
    const int blocks = outer;

    // Prepare dimension information
    int64_t h_sizes[32], h_strides[32];
    for (int i = 0; i < ndim; i++) {
        h_sizes[i] = x.size(i);
        h_strides[i] = x.stride(i);
    }

    int64_t *d_sizes, *d_strides;
    cudaMalloc(&d_sizes, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_sizes, h_sizes, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, h_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hybrid_reverse_cumsum", ([&] {
        warp_optimized_reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
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