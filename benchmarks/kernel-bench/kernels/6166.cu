#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    __shared__ scalar_t shared_input[128 * 128];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int BLOCK_SIZE = 128;
    
    // Calculate output position
    const int total = N * C * outH * outW;
    const int index = bid * BLOCK_SIZE + tid;
    if (index >= total) return;
    
    const int w_out = index % outW;
    const int h_out = (index / outW) % outH;
    const int c = (index / (outW * outH)) % C;
    const int n = index / (outW * outH * C);
    
    // Calculate input window boundaries
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;
    const int h_end = h_start + kernel_size;
    const int w_end = w_start + kernel_size;
    
    // Calculate tile dimensions
    const int tile_h = kernel_size + (BLOCK_SIZE / kernel_size);
    const int tile_w = kernel_size;
    
    scalar_t sum_val = 0;
    
    // Load input tile into shared memory
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            if (h >= 0 && h < H && w >= 0 && w < W) {
                const int shared_idx = (h - h_start) * tile_w + (w - w_start);
                if (shared_idx < tile_h * tile_w) {
                    shared_input[shared_idx] = input[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute average using shared memory
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            const int h = h_start + i;
            const int w = w_start + j;
            if (h >= 0 && h < H && w >= 0 && w < W) {
                const int shared_idx = i * tile_w + j;
                sum_val += shared_input[shared_idx];
            }
        }
    }
    
    output[index] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
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

    const int threads = 128;
    const int blocks = (N * C * outH * outW + threads - 1) / threads;
    
    // Calculate shared memory size
    const int tile_h = kernel_size + (threads / kernel_size);
    const int tile_w = kernel_size;
    const int shared_mem_size = tile_h * tile_w * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        avg_pool2d_forward_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}