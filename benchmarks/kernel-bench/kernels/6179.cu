#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function for average pooling
__global__ void avg_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * outH * outW;
    if (index >= total) return;

    int w_out = index % outW;
    int h_out = (index / outW) % outH;
    int c = index / (outH * outW);

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float sum_val = 0;
    int count = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                sum_val += input[(c * H + h_in) * W + w_in];
                count++;
            }
        }
    }
    output[index] = sum_val / count;
}

// Function to perform average pooling using CUDA streams
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
    const int blocks = (C * outH * outW + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward_kernel", ([&] {
        const float* input_data = x_cont.data_ptr<float>();
        float* output_data = out.data_ptr<float>();

        for (int n = 0; n < N; ++n) {
            const float* sample_input = input_data + n * C * H * W;
            float* sample_output = output_data + n * C * outH * outW;


            avg_pool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
                sample_input, sample_output,
                C, H, W, outH, outW,
                kernel_size, stride, padding
            );
        }
    }));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}