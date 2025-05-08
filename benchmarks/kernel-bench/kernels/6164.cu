#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes one output element per block. Threads within the block cooperate
// to sum over the pooling window. Partial sums are accumulated using atomicAdd on a shared
// memory variable. This leverages fast shared memory atomics while avoiding global atomics,
// reducing contention and handling race conditions only where necessary.

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_atomic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int outH,
    int outW,
    int kernel_size,
    int stride,
    int padding) {

    int total_outputs = N * C * outH * outW;
    int out_index = blockIdx.x; // One block computes one output element
    if (out_index >= total_outputs) return;

    // Decode the output index into (n, c, h_out, w_out)
    int w_out = out_index % outW;
    int h_out = (out_index / outW) % outH;
    int c = (out_index / (outW * outH)) % C;
    int n = out_index / (outW * outH * C);

    // Calculate the starting indices of the pooling window in the input
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    int pool_area = kernel_size * kernel_size;

    // Shared memory variable to accumulate the sum
    __shared__ scalar_t shared_sum[32];
    if (threadIdx.x == 0) {
        shared_sum = scalar_t(0);
    }
    __syncthreads();

    // Each thread processes a portion of the pooling window
    scalar_t local_sum = scalar_t(0);
    for (int idx = threadIdx.x; idx < pool_area; idx += blockDim.x) {
        int i = idx / kernel_size;
        int j = idx % kernel_size;
        int h_in = h_start + i;
        int w_in = w_start + j;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int input_index = ((n * C + c) * H + h_in) * W + w_in;
            local_sum += input[input_index];
        }
    }

    // Use atomic operation on shared memory to accumulate partial sums
    atomicAdd(&shared_sum, local_sum);
    __syncthreads();

    // Single thread writes the final averaged value to global memory
    if (threadIdx.x == 0) {
        output[out_index] = shared_sum / static_cast<scalar_t>(pool_area);
    }
}


torch::Tensor avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto x_cont = x.contiguous();
    auto options = x.options();
    auto out = torch::empty({N, C, outH, outW}, options);

    int total_output = N * C * outH * outW;
    int pool_area = kernel_size * kernel_size;
    // Choose block size: if pool_area is small, use that, otherwise cap at 256 threads.
    int block_threads = pool_area < 256 ? pool_area : 256;

    // Launch one block per output element
    dim3 grid(total_output);
    dim3 block(block_threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_atomic", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_atomic<scalar_t><<<grid, block>>>(
            input_data,
            output_data,
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA) with block-level atomic reduction");
}
