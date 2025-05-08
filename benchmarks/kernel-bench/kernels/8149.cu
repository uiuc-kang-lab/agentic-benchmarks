#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(shared_memory + sizeof(scalar_t) * blockDim.x);

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
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

        scalar_t sum = 0;

        // Process kernel window with shared memory
        for (int kh = 0; kh < kernel_h; kh++) {
            const int h_in = (oh - kh * dilation + padding) / stride;
            if ((oh - kh * dilation + padding) % stride != 0 || h_in < 0 || h_in >= in_height) continue;

            for (int kw = 0; kw < kernel_w; kw++) {
                const int w_in = (ow - kw * dilation + padding) / stride;
                if ((ow - kw * dilation + padding) % stride != 0 || w_in < 0 || w_in >= in_width) continue;

                // Load input and weight data into shared memory
                for (int ic = threadIdx.x; ic < in_channels_per_group; ic += blockDim.x) {
                    if (ic < in_channels_per_group) {
                        const int input_idx = b * in_channels * in_height * in_width +
                                            (ic_start + ic) * in_height * in_width +
                                            h_in * in_width + w_in;
                        const int weight_idx = (ic_start + ic) * out_channels_per_group * kernel_h * kernel_w +
                                             oc_group * kernel_h * kernel_w +
                                             kh * kernel_w + kw;
                        
                        shared_input[ic] = input[input_idx];
                        shared_weight[ic] = weight[weight_idx];
                    }
                }
                __syncthreads();

                // Compute partial sums within each warp
                scalar_t warp_sum = 0;
                for (int ic = lane_id; ic < in_channels_per_group; ic += warp_size) {
                    if (ic < in_channels_per_group) {
                        warp_sum += shared_input[ic] * shared_weight[ic];
                    }
                }

                // Warp-level reduction using shuffle operations
                #pragma unroll
                for (int offset = warp_size/2; offset > 0; offset /= 2) {
                    warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
                }

                // First thread in each warp has the result
                if (lane_id == 0) {
                    sum += warp_sum;
                }
                __syncthreads();
            }
        }

        // Add bias and write final result
        if (threadIdx.x < warp_size) {
            scalar_t final_sum = __shfl_sync(0xffffffff, sum, 0);
            if (lane_id == 0) {
                output[idx] = final_sum + (bias != nullptr ? bias[oc] : 0);
            }
        }
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

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    const int shared_mem_size = sizeof(float) * threads * 2; // Space for both input and weight data

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda_shared", ([&] {
        conv_transpose2d_kernel_shared<scalar_t><<<blocks, threads, shared_mem_size>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Transposed 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}