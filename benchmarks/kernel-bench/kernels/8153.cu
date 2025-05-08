#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modularized kernel refactored with shared memory optimization for better performance

template <typename scalar_t>
__device__ scalar_t compute_convolution_shared(
    const scalar_t* __restrict__ input_group,
    const scalar_t* __restrict__ weight_group,
    const int oh,
    const int ow,
    const int in_channels_per_group,
    const int in_height,
    const int in_width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int dilation,
    const int out_channels_per_group
) {
    scalar_t sum = 0;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_in_temp = oh - kh * dilation;
        if (h_in_temp % stride != 0) continue;
        int h_in = h_in_temp / stride;
        if (h_in < 0 || h_in >= in_height) continue;

        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_in_temp = ow - kw * dilation;
            if (w_in_temp % stride != 0) continue;
            int w_in = w_in_temp / stride;
            if (w_in < 0 || w_in >= in_width) continue;

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                sum += input_group[(ic) * (in_height * in_width) + h_in * in_width + w_in] 
                     * weight_group[(ic) * (out_channels_per_group * kernel_h * kernel_w) 
                                    + kh * kernel_w + kw];
            }
        }
    }
    return sum;
}

// Kernel function utilizing the device function

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_shared(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    extern __shared__ char shared_data[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_data);
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(shared_data) + in_channels * in_height * in_width;

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int gridStride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridStride) {
        int n = idx;
        const int ow = n % out_width;
        n /= out_width;
        const int oh = n % out_height;
        n /= out_height;
        const int oc = n % out_channels;
        n /= out_channels;
        const int b = n;

        if (b >= batch_size) continue;

        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

        // Load input and weight into shared memory
        if (threadIdx.x < in_channels_per_group * in_height * in_width) {
            shared_input[threadIdx.x] = input[ic_start * in_height * in_width + threadIdx.x];
        }

        if (threadIdx.x < in_channels_per_group * out_channels_per_group * kernel_h * kernel_w) {
            shared_weight[threadIdx.x] = weight[ic_start * out_channels_per_group * kernel_h * kernel_w + threadIdx.x];
        }

        __syncthreads();

        val += compute_convolution_shared(
            shared_input, shared_weight, oh, ow,
            in_channels_per_group, in_height, in_width, kernel_h, kernel_w,
            stride, dilation, out_channels_per_group
        );

        output[idx] = val;

        // Ensure all threads are synchronized
        __syncthreads();
    }
}


// Forward function that sets up kernel parameters

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

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = output.numel();
    constexpr int THREADS = 256;
    const int BLOCKS = (total_elements + THREADS - 1) / THREADS;
    const size_t shared_mem_size = (in_channels * in_height * in_width + in_channels * out_channels / groups * kernel_h * kernel_w) * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda_shared", ([&] {
        conv_transpose2d_kernel_shared<scalar_t><<<BLOCKS, THREADS, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

// Pybind the forward function

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Optimized Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
