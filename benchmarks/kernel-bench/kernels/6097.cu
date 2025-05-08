#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory for intra-block reductions and warp-level reductions
// to optimize the 2D average pooling operation.

template <typename scalar_t>
__global__ void shared_memory_avg_pool2d_kernel(
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
    extern __shared__ scalar_t shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int out_x = blockIdx.x * blockDim.x + tx;
    int out_y = blockIdx.y * blockDim.y + ty;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    if (out_x >= outW || out_y >= outH)
        return;

    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;

    scalar_t sum = 0;
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            int x = in_x_start + kx;
            int y = in_y_start + ky;
            if (x >= 0 && x < W && y >= 0 && y < H) {
                int input_idx = ((n * C + c) * H + y) * W + x;
                sum += input[input_idx];
            }
        }
    }

    shared_mem[tid] = sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction (ensure full warp participation)
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            shared_mem[tid] += __shfl_down_sync(0xFFFFFFFF, shared_mem[tid], offset);
        }
    }

    if (tid == 0) {
        output[((n * C + c) * outH + blockIdx.y) * outW + blockIdx.x] = shared_mem[0] / (kernel_size * kernel_size);
    }
}

// Forward function exposed to PyTorch

torch::Tensor shared_memory_avg_pool2d_forward(
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
    
    // Use a 32x8 block size for spatial dimensions
    dim3 threads(32, 8);
    dim3 blocks(
        (outW + threads.x - 1) / threads.x,
        (outH + threads.y - 1) / threads.y,
        N * C
    );
    
    const size_t shared_mem_size = threads.x * threads.y * sizeof(scalar_t);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_memory_avg_pool2d_kernel", ([&] {
        shared_memory_avg_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &shared_memory_avg_pool2d_forward, "Shared Memory 2D Average Pooling forward (CUDA)");
}
