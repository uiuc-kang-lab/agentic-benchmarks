#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_per_sample_kernel(
    const scalar_t* __restrict__ input_sample,
    scalar_t* __restrict__ output_sample,
    int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    int total = C * outH * outW;
    int kernel_area = kernel_size * kernel_size;
    scalar_t inv_kernel_area = static_cast<scalar_t>(1) / static_cast<scalar_t>(kernel_area);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {

    int w_out = index % outW;
    int h_out = (index / outW) % outH;
    int c = index / (outH * outW);

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    scalar_t sum_val = scalar_t(0);
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            if (h_in >=0 && h_in < H && w_in >=0 && w_in < W) {
                sum_val += input_sample[(c * H + h_in) * W + w_in];
            }
        }
    }
    output_sample[index] = sum_val / static_cast<scalar_t>(kernel_size*kernel_size);
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

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "avg_pool2d_forward", ([&] {
        const scalar_t* input_data = x_cont.data_ptr<scalar_t>();
        scalar_t* output_data = out.data_ptr<scalar_t>();

        for (int n = 0; n < N; ++n) {
            int stream_id = n % num_streams;
            const scalar_t* sample_input = input_data + n * C * H * W;
            scalar_t* sample_output = output_data + n * C * outH * outW;
            
            int total_per_sample = C * outH * outW;
            const int threads = 256;
            const int blocks = (total_per_sample + threads - 1) / threads;

            avg_pool2d_per_sample_kernel<<<blocks, threads, 0, streams[stream_id]>>>(
                sample_input, sample_output,
                C, H, W, outH, outW,
                kernel_size, stride, padding
            );
        }
    }));

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool2d_forward, "2D Average Pooling forward (CUDA)");
}