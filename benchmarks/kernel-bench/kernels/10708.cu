#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Template parameter BLOCK_SIZE allows for compile-time optimization of block size
template <typename scalar_t, int BLOCK_SIZE>
__global__ void block_tuned_reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                                   scalar_t* __restrict__ output,
                                                   int64_t n) {
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    __shared__ scalar_t block_prefix[BLOCK_SIZE/32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int64_t row_offset = row * n;

    // Load data in reverse order
    scalar_t thread_val = 0;
    if (tid < n) {
        thread_val = __ldg(&input[row_offset + (n - 1 - tid)]);
    }
    shared_data[tid] = thread_val;
    __syncthreads();

    // Perform warp-level scan
    scalar_t warp_sum = thread_val;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t temp = __shfl_up_sync(0xffffffff, warp_sum, offset);
        if (lane_id >= offset) {
            warp_sum += temp;
        }
    }

    // Store warp results
    if (lane_id == 31) {
        block_prefix[warp_id] = warp_sum;
    }
    __syncthreads();

    // Compute warp offsets
    if (warp_id == 0 && tid < (BLOCK_SIZE/32)) {
        scalar_t running_sum = 0;
        for (int i = 0; i < tid; i++) {
            running_sum += block_prefix[i];
        }
        block_prefix[tid] = running_sum;
    }
    __syncthreads();

    // Add warp prefix to each thread's value
    if (tid < n) {
        scalar_t final_sum = warp_sum;
        if (warp_id > 0) {
            final_sum += block_prefix[warp_id];
        }
        output[row_offset + (n - 1 - tid)] = final_sum;
    }
}

template <typename scalar_t>
void launch_kernel_with_block_size(const scalar_t* input, scalar_t* output,
                                 int64_t n, int64_t outer, cudaStream_t stream) {
    dim3 grid(outer);
    
    if (n <= 128) {
        dim3 block(128);
        block_tuned_reverse_cumsum_kernel<scalar_t, 128><<<grid, block, 0, stream>>>(input, output, n);
    }
    else if (n <= 256) {
        dim3 block(256);
        block_tuned_reverse_cumsum_kernel<scalar_t, 256><<<grid, block, 0, stream>>>(input, output, n);
    }
    else if (n <= 512) {
        dim3 block(512);
        block_tuned_reverse_cumsum_kernel<scalar_t, 512><<<grid, block, 0, stream>>>(input, output, n);
    }
    else {
        dim3 block(1024);
        block_tuned_reverse_cumsum_kernel<scalar_t, 1024><<<grid, block, 0, stream>>>(input, output, n);
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

    if (dim == ndim - 1 && n <= 1024) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "block_tuned_reverse_cumsum_kernel", ([&] {
            launch_kernel_with_block_size<scalar_t>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n,
                outer,
                stream);
        }));
    } else {
        auto x_flipped = x.flip(dim);
        auto cumsum = x_flipped.cumsum(dim);
        output = cumsum.flip(dim);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Block-size tuned reverse cumulative sum (CUDA)");
}