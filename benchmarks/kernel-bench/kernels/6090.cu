#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses shared memory and warp-level primitives for intra-block reduction
// Each block computes one output element (one pooling window average).

template <typename scalar_t>
__global__ void shared_reduction_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int outH,
    const int outW,
    const int kernel_size,
    const int stride,
    const int padding
) {
    // Each block handles one output element.
    int out_index = blockIdx.x;
    int total = N * C * outH * outW;
    if (out_index >= total) return;

    // Map the linear index to (n, c, h_out, w_out)
    int w_out = out_index % outW;
    int tmp = out_index / outW;
    int h_out = tmp % outH;
    tmp = tmp / outH;
    int c = tmp % C;
    int n = tmp / C;

    // Calculate the starting indices of the pooling window
    int in_h_start = h_out * stride - padding;
    int in_w_start = w_out * stride - padding;
    int pool_size = kernel_size * kernel_size;

    // Each thread in the block will accumulate a partial sum over a subset of the pooling window
    scalar_t partial_sum = 0;
    for (int idx = threadIdx.x; idx < pool_size; idx += blockDim.x) {
        int ky = idx / kernel_size;
        int kx = idx % kernel_size;
        int in_y = in_h_start + ky;
        int in_x = in_w_start + kx;
        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
            int input_index = ((n * C + c) * H + in_y) * W + in_x;
            partial_sum += input[input_index];
        }
    }

    // Reduction using shared memory and warp-level primitives
    // We assume blockDim.x is set to 32
    __shared__ scalar_t shared_data[32];
    shared_data[threadIdx.x] = partial_sum;
    __syncthreads();

    // Use warp-level reduction (assuming blockDim.x <= warpSize)
    scalar_t sum_val = shared_data[threadIdx.x];
    // Full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    }
    
    if (threadIdx.x == 0) {
        // Write averaged result to output
        output[out_index] = sum_val / static_cast<scalar_t>(pool_size);
    }
}

// Host function for launching the kernel

torch::Tensor shared_reduction_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor.");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    const int total_outputs = N * C * outH * outW;

    auto x_contig = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);

    // Launch one block per output element and 32 threads per block
    const int threads = 32;
    const int blocks = total_outputs;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_reduction_avg_pool2d_kernel", ([&] {
        shared_reduction_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W, outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_reduction_avg_pool2d_forward, "Shared Reduction 2D Average Pooling forward (CUDA)");
}
