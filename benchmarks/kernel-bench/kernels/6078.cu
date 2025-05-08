#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

template <typename scalar_t>
__global__ void streamed_avg_pool2d_kernel(
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
    int padding,
    int chunk_start,
    int chunk_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_end = min(chunk_start + chunk_size, N * C * outH * outW);
    
    if (tid + chunk_start >= chunk_end) return;

    int idx = tid + chunk_start;
    int w_out = idx % outW;
    int h_out = (idx / outW) % outH;
    int c = (idx / (outW * outH)) % C;
    int n = idx / (outW * outH * C);

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t sum_val = scalar_t(0);
    #pragma unroll
    for (int i = 0; i < kernel_size; i++) {
        #pragma unroll
        for (int j = 0; j < kernel_size; j++) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                sum_val += input[((n * C + c) * H + h_in) * W + w_in];
            }
        }
    }
    output[idx] = sum_val / static_cast<scalar_t>(kernel_size * kernel_size);
}

torch::Tensor streamed_avg_pool2d_forward(
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
    auto output = torch::empty({N, C, outH, outW}, options);

    const int total_elements = N * C * outH * outW;
    const int chunk_size = (total_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;
    const int blocks_per_chunk = (chunk_size + threads - 1) / threads;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "streamed_avg_pool2d_kernel", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        for (int i = 0; i < NUM_STREAMS; i++) {
            int chunk_start = i * chunk_size;
            streamed_avg_pool2d_kernel<scalar_t><<<blocks_per_chunk, threads, 0, streams[i]>>>(
                input_data,
                output_data,
                N, C, H, W,
                outH, outW,
                kernel_size, stride, padding,
                chunk_start,
                chunk_size
            );
        }
    }));

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_avg_pool2d_forward, "Streamed 2D Average Pooling forward (CUDA)");
}