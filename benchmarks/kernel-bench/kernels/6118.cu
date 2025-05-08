#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 2D average pooling kernel using shared memory and warp-level reduction

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel_shared(
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
    int padding
) {
    // Each block processes one output element
    int out_idx = blockIdx.x;
    if (out_idx >= N * C * outH * outW) return;

    // Decode output indices
    int w_out = out_idx % outW;
    int h_out = (out_idx / outW) % outH;
    int c = (out_idx / (outW * outH)) % C;
    int n = out_idx / (outW * outH * C);

    // Calculate the top-left corner of the pooling window in the input
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    // Total number of elements in the pooling window
    int total_elements = kernel_size * kernel_size;

    // Each thread in the block computes partial sum over a subset of pooling window elements
    scalar_t thread_sum = static_cast<scalar_t>(0);
    
    // Process rows in the pooling window
    for (int i = 0; i < kernel_size; i++) {
        int h_in = h_start + i;
        if (h_in >= 0 && h_in < H) {
            // Process elements within a row collaboratively among threads
            for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
                int w_in = w_start + j;
                if (w_in >= 0 && w_in < W) {
                    thread_sum += input[((n * C + c) * H + h_in) * W + w_in];
                }
            }
        }
    }

    // Intra-warp reduction using warp shuffle primitives
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's first thread holds the partial sum for that warp
    __shared__ scalar_t shared_sum[32];  // Assuming a maximum of 32 warps per block
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction of warp-level partial sums
    scalar_t block_sum = static_cast<scalar_t>(0);
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        block_sum = shared_sum[threadIdx.x];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
    }

    // The first thread in the block writes the output
    if (threadIdx.x == 0) {
        output[out_idx] = block_sum / static_cast<scalar_t>(total_elements);
    }
}


// Host function

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

    // Launch one block per output element using a fixed number of threads
    const int threads = 256;
    const int blocks = N * C * outH * outW;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_shared", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel_shared<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward with shared memory reduction (CUDA)");
}
