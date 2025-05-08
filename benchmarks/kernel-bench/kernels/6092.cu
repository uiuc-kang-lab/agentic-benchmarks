#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses shared memory and warp-level primitives to perform efficient reductions.

// Warp size
#define WARP_SIZE 32

// Shared memory reduction kernel for 2D average pooling

template <typename scalar_t>
__global__ void shared_memory_warp_reduction_avg_pool2d_kernel(
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
    // Thread index
    const int tid = threadIdx.x;
    // Map threads in a 2D block to output spatial dimensions
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Use blockIdx.z to consider the (N, C) dimension
    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc % C;

    if (out_x >= outW || out_y >= outH)
        return;

    // Compute the top-left corner of the pooling window in the input
    const int in_x_start = out_x * stride - padding;
    const int in_y_start = out_y * stride - padding;
    const int in_x_end = in_x_start + kernel_size;
    const int in_y_end = in_y_start + kernel_size;

    scalar_t sum = scalar_t(0);
    
    // Shared memory for block reduction
    __shared__ scalar_t shared_sum[WARP_SIZE];

    for (int ky = in_y_start; ky < in_y_end; ++ky) {
        #pragma unroll
        for (int kx = in_x_start; kx < in_x_end; ++kx) {
            if (ky >= 0 && ky < H && kx >= 0 && kx < W) {
                int index_in = ((n * C + c) * H + ky) * W + kx;
                sum += input[index_in];
            }
        }
    }

    // Save the sum in shared memory
    shared_sum[tid] = sum;
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        output[((n * C + c) * outH + out_y) * outW + out_x] = shared_sum[0] / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

// Host function to launch the kernel

torch::Tensor shared_memory_warp_reduction_avg_pool2d_forward(
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
    
    auto x_cont = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);
    
    // Use a 2D block for spatial dimensions and gridDim.z for the combined (N, C) dimension
    dim3 threads(32, 1);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_memory_warp_reduction_avg_pool2d_kernel", ([&] {
        shared_memory_warp_reduction_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            x_cont.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W,
            outH, outW,
            kernel_size, stride, padding
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_warp_reduction_avg_pool2d_forward, "Shared Memory and Warp Reduction 2D Average Pooling forward (CUDA)");
}
