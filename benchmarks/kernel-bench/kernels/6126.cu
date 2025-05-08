#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes 2D average pooling by splitting the pooling window
// across multiple blocks in the y-dimension. Each block computes a partial sum
// for the same output element, and then a shared memory reduction is performed
// so that only one atomicAdd is issued per block. This minimizes atomic usage
// on global memory, reducing contention, while ensuring the correct result.

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
    int padding
) {
    // Each block in grid.x corresponds to one output element
    int out_index = blockIdx.x;
    if (out_index >= N * C * outH * outW) return;

    // Decode output index into (n, c, h_out, w_out)
    int w_out = out_index % outW;
    int h_out = (out_index / outW) % outH;
    int c = (out_index / (outW * outH)) % C;
    int n = out_index / (outW * outH * C);

    // Calculate the top-left corner of the pooling window in the input
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    int pool_area = kernel_size * kernel_size;

    // Use gridDim.y to partition the pooling window among blocks:
    // Each block in the y-dimension processes a chunk of pooling indices.
    int pool_index = blockIdx.y * blockDim.x + threadIdx.x;
    scalar_t local_sum = static_cast<scalar_t>(0);
    if (pool_index < pool_area) {
        int i = pool_index / kernel_size;
        int j = pool_index % kernel_size;
        int h_in = h_start + i;
        int w_in = w_start + j;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            local_sum = input[((n * C + c) * H + h_in) * W + w_in];
        }
    }

    // Reduction within the block using shared memory to accumulate partial sums
    extern __shared__ scalar_t shmem[];
    shmem[threadIdx.x] = local_sum;
    __syncthreads();

    // Standard reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 scales the block's partial sum and atomically adds it to global output
    if (threadIdx.x == 0) {
        // Scale by the pooling area so that final result is average
        scalar_t scaled = shmem[0] / static_cast<scalar_t>(pool_area);
        // Atomic add to resolve race conditions across different chunks
        atomicAdd(&output[out_index], scaled);
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
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto options = x.options();
    // Initialize output to zero since we are accumulating partial results
    auto out = torch::zeros({N, C, outH, outW}, options);

    int pool_area = kernel_size * kernel_size;
    // Choose a block size; if pool_area is small, extra threads will do no work
    int threads = 256;
    // Number of chunks in the pooling window processed per output element
    int chunks = (pool_area + threads - 1) / threads;

    // Grid: x-dim over each output element, y-dim over chunks of the pooling window
    dim3 blocks;
    blocks.x = N * C * outH * outW;
    blocks.y = chunks;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel_atomic", ([&] {
        avg_pool2d_forward_kernel_atomic<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward with minimized atomic operations (CUDA)");
}
