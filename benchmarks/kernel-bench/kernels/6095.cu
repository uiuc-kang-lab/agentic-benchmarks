#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void shared_warp_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N, const int C,
    const int H, const int W,
    const int outH, const int outW,
    const int kernel_size,
    const int stride,
    const int padding
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    
    // Calculate output indices
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y;
    const int n = blockIdx.z / C;
    const int c = blockIdx.z % C;

    if (out_x >= outW) return;

    const int in_y_start = out_y * stride - padding;
    const int in_x_start = out_x * stride - padding;
    
    // Initialize accumulator
    scalar_t sum = 0;

    // First phase: threads accumulate their assigned elements
    #pragma unroll
    for (int ky = 0; ky < kernel_size; ++ky) {
        const int in_y = in_y_start + ky;
        if (in_y >= 0 && in_y < H) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_x = in_x_start + kx;
                if (in_x >= 0 && in_x < W) {
                    sum += input[((n * C + c) * H + in_y) * W + in_x];
                }
            }
        }
    }

    // Store partial sum in shared memory
    shared_data[threadIdx.x] = sum;
    __syncthreads();

    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes result
    if (lane_id == 0 && out_x < outW) {
        const int out_idx = ((n * C + c) * outH + out_y) * outW + out_x;
        output[out_idx] = sum / static_cast<scalar_t>(kernel_size * kernel_size);
    }
}

torch::Tensor shared_warp_avg_pool2d_forward(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor");
    
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    
    const int outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int outW = (W + 2 * padding - kernel_size) / stride + 1;
    
    auto x_cont = x.contiguous();
    auto options = x.options();
    auto output = torch::empty({N, C, outH, outW}, options);

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int shared_memory_size = threads_per_block * sizeof(float);
    
    dim3 threads(threads_per_block);
    dim3 blocks(
        (outW + threads_per_block - 1) / threads_per_block,
        outH,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_warp_avg_pool2d_kernel", ([&] {
        shared_warp_avg_pool2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &shared_warp_avg_pool2d_forward, "Shared Memory and Warp-Level 2D Average Pooling forward (CUDA)");
}