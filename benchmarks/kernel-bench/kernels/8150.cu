#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int CONV_PARAMS[8];  // [in_height, in_width, kernel_h, kernel_w, stride, padding, dilation, groups]

template <typename scalar_t>
__device__ __forceinline__ bool check_bounds(
    const int oh, const int ow,
    const int kh, const int kw,
    int& h_in, int& w_in
) {
    int h_temp = oh - kh * CONV_PARAMS[6] + CONV_PARAMS[5];  // dilation, padding
    int w_temp = ow - kw * CONV_PARAMS[6] + CONV_PARAMS[5];

    if (h_temp % CONV_PARAMS[4] != 0 || w_temp % CONV_PARAMS[4] != 0)  // stride check
        return false;

    h_in = h_temp / CONV_PARAMS[4];
    w_in = w_temp / CONV_PARAMS[4];

    return (h_in >= 0 && h_in < CONV_PARAMS[0] &&    // in_height
            w_in >= 0 && w_in < CONV_PARAMS[1]);      // in_width
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t fetch_and_multiply(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int input_idx,
    const int weight_idx
) {
    return input[input_idx] * weight[weight_idx];
}

template <typename scalar_t>
__device__ __forceinline__ int compute_input_index(
    const int batch_idx,
    const int ic,
    const int h_in,
    const int w_in,
    const int in_channels,
    const int in_height,
    const int in_width
) {
    return batch_idx * (in_channels * in_height * in_width) +
           ic * (in_height * in_width) +
           h_in * in_width + w_in;
}

template <typename scalar_t>
__device__ __forceinline__ int compute_weight_index(
    const int ic,
    const int oc,
    const int kh,
    const int kw,
    const int out_channels_per_group,
    const int kernel_h,
    const int kernel_w
) {
    return ic * (out_channels_per_group * kernel_h * kernel_w) +
           oc * (kernel_h * kernel_w) +
           kh * kernel_w + kw;
}

template <typename scalar_t>
__device__ scalar_t compute_convolution(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int batch_idx,
    const int oc_group,
    const int oh,
    const int ow,
    const int in_channels_per_group,
    const int ic_start,
    const int out_channels_per_group
) {
    scalar_t sum = 0;

    #pragma unroll 4
    for (int kh = 0; kh < CONV_PARAMS[2]; ++kh) {  // kernel_h
        #pragma unroll 4
        for (int kw = 0; kw < CONV_PARAMS[3]; ++kw) {  // kernel_w
            int h_in, w_in;
            if (!check_bounds(oh, ow, kh, kw, h_in, w_in)) continue;

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                const int input_idx = compute_input_index<scalar_t>(
                    batch_idx, ic_start + ic, h_in, w_in,
                    in_channels_per_group * CONV_PARAMS[7],  // groups
                    CONV_PARAMS[0], CONV_PARAMS[1]           // in_height, in_width
                );

                const int weight_idx = compute_weight_index<scalar_t>(
                    ic, oc_group, kh, kw,
                    out_channels_per_group,
                    CONV_PARAMS[2], CONV_PARAMS[3]  // kernel_h, kernel_w
                );

                sum += fetch_and_multiply(input, weight, input_idx, weight_idx);
            }
        }
    }
    return sum;
}

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_modular_opt(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int out_height,
    const int out_width
) {
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        
        const int ow = idx % out_width;
        const int oh = (idx / out_width) % out_height;
        const int oc = (idx / (out_width * out_height)) % out_channels;
        const int b = idx / (out_width * out_height * out_channels);

        const int out_channels_per_group = out_channels / CONV_PARAMS[7];  // groups
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / CONV_PARAMS[7];
        const int ic_start = g * in_channels_per_group;

        scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
        
        val += compute_convolution(
            input, weight, b, oc_group, oh, ow,
            in_channels_per_group, ic_start, out_channels_per_group
        );

        output[idx] = val;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    // Copy parameters to constant memory
    int h_params[8] = {in_height, in_width, kernel_h, kernel_w, stride, padding, dilation, groups};
    cudaMemcpyToSymbol(CONV_PARAMS, h_params, sizeof(int) * 8);

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = output.numel();
    constexpr int THREADS = 256;
    const int BLOCKS = (total_elements + THREADS - 1) / THREADS;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda_modular_opt", ([&] {
        conv_transpose2d_kernel_modular_opt<scalar_t><<<BLOCKS, THREADS>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Optimized Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}