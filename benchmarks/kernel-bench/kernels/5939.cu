#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[7]; // [kernel_size, stride, padding, dilation, input_d, input_h, input_w]

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t* indices,
    const int batch_size,
    const int channels,
    const int output_d, const int output_h, const int output_w) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch_size * channels * output_d * output_h * output_w) return;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    const int d_start = d_out * const_params[1] - const_params[2];
    const int h_start = h_out * const_params[1] - const_params[2];
    const int w_start = w_out * const_params[1] - const_params[2];

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int k_d = 0; k_d < const_params[0]; k_d++) {
        const int d_in = d_start + k_d * const_params[3];
        if (d_in < 0 || d_in >= const_params[4]) continue;

        #pragma unroll
        for (int k_h = 0; k_h < const_params[0]; k_h++) {
            const int h_in = h_start + k_h * const_params[3];
            if (h_in < 0 || h_in >= const_params[5]) continue;

            #pragma unroll
            for (int k_w = 0; k_w < const_params[0]; k_w++) {
                const int w_in = w_start + k_w * const_params[3];
                if (w_in < 0 || w_in >= const_params[6]) continue;

                const int input_idx = ((b * channels + c) * const_params[4] + d_in) * const_params[5] * const_params[6] +
                                    h_in * const_params[6] + w_in;
                const scalar_t val = input[input_idx];

                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

void set_constant_params(int kernel_size, int stride, int padding, int dilation, int input_d, int input_h, int input_w) {
    int h_params[7] = {kernel_size, stride, padding, dilation, input_d, input_h, input_w};
    cudaMemcpyToSymbol(const_params, h_params, sizeof(int) * 7);
}

torch::Tensor max_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {
    
    auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    const int output_d = ceil_mode ? 
        ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_h = ceil_mode ?
        ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_w = ceil_mode ?
        ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ? 
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    set_constant_params(kernel_size, stride, padding, dilation, input_d, input_h, input_w);

    const int threads = 256;
    const int elements = batch_size * channels * output_d * output_h * output_w;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            channels,
            output_d, output_h, output_w);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward (CUDA)");
}