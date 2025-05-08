#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
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
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(shared_memory + sizeof(scalar_t) * blockDim.x);

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;

    // Unravel output index
    int n = idx;
    const int ow = n % out_width;
    n /= out_width;
    const int oh = n % out_height;
    n /= out_height;
    const int oc = n % out_channels;
    n /= out_channels;
    const int b = n;

    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    // Initialize output value
    scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    // Process input in tiles to maximize shared memory usage
    const int TILE_SIZE = 16;
    const int input_channel_stride = in_height * in_width;
    const int weight_channel_stride = kernel_h * kernel_w;

    for (int ic_tile = 0; ic_tile < in_channels_per_group; ic_tile += TILE_SIZE) {
        const int ic_end = min(ic_tile + TILE_SIZE, in_channels_per_group);
        
        // Load weight tile into shared memory
        for (int ic_offset = tid; ic_offset < (ic_end - ic_tile) * kernel_h * kernel_w; ic_offset += blockDim.x) {
            int ic_local = ic_offset / (kernel_h * kernel_w);
            int k_idx = ic_offset % (kernel_h * kernel_w);
            int ic = ic_tile + ic_local;
            if (ic < in_channels_per_group) {
                shared_weight[ic_offset] = weight[
                    (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                    oc_group * weight_channel_stride +
                    k_idx
                ];
            }
        }
        __syncthreads();

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in_base = oh - kh * dilation + padding;
            if (h_in_base % stride != 0) continue;
            int h_in = h_in_base / stride;
            if (h_in < 0 || h_in >= in_height) continue;

            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in_base = ow - kw * dilation + padding;
                if (w_in_base % stride != 0) continue;
                int w_in = w_in_base / stride;
                if (w_in < 0 || w_in >= in_width) continue;

                // Load input values for current position into shared memory
                for (int ic = ic_tile; ic < ic_end; ++ic) {
                    if (tid == 0) {  // Only one thread loads each input value
                        shared_input[ic - ic_tile] = input[
                            b * in_channels * input_channel_stride +
                            (ic_start + ic) * input_channel_stride +
                            h_in * in_width +
                            w_in
                        ];
                    }
                }
                __syncthreads();

                // Compute partial sum for this tile
                for (int ic = ic_tile; ic < ic_end; ++ic) {
                    val += shared_input[ic - ic_tile] *
                           shared_weight[(ic - ic_tile) * kernel_h * kernel_w + kh * kernel_w + kw];
                }
                __syncthreads();
            }
        }
    }

    output[idx] = val;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    const int shared_mem_size = sizeof(float) * (threads + 16 * kernel_h * kernel_w); // For input and weight tiles

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}