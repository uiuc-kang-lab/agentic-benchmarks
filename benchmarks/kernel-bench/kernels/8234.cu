#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int total_warps = warps_per_block * gridDim.x;
    
    // Global warp index
    const int global_warp_idx = blockIdx.x * warps_per_block + warp_id;
    
    // Total output elements
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    // Process elements per warp
    for (int idx = global_warp_idx * warp_size + lane_id; 
         idx < total_elements; 
         idx += total_warps * warp_size) {
        
        // Calculate output indices
        const int ow = idx % out_width;
        const int oh = (idx / out_width) % out_height;
        const int oc = (idx / (out_width * out_height)) % out_channels;
        const int b = idx / (out_width * out_height * out_channels);
        
        if (b >= batch_size) continue;

        const int out_channels_per_group = out_channels / groups;
        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int in_channels_per_group = in_channels / groups;
        const int ic_start = g * in_channels_per_group;

        scalar_t val = 0;

        // Pre-compute base addresses
        const scalar_t* input_base = input + b * in_channels * in_height * in_width;
        const scalar_t* weight_base = weight + (ic_start) * (out_channels_per_group * kernel_h * kernel_w) + 
                                    oc_group * kernel_h * kernel_w;

        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int h_in = (oh - kh*dilation + padding) / stride;
            if ((oh - kh*dilation + padding) % stride != 0) continue;
            if (h_in < 0 || h_in >= in_height) continue;

            #pragma unroll
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_in = (ow - kw*dilation + padding) / stride;
                if ((ow - kw*dilation + padding) % stride != 0) continue;
                if (w_in < 0 || w_in >= in_width) continue;

                // Each thread in warp processes different input channels
                scalar_t thread_sum = 0;
                
                #pragma unroll
                for (int ic = lane_id; ic < in_channels_per_group; ic += warp_size) {
                    const scalar_t x_val = input_base[(ic_start + ic) * in_height * in_width + 
                                                    h_in * in_width + w_in];
                    
                    const scalar_t w_val = weight_base[ic * (out_channels_per_group * kernel_h * kernel_w) + 
                                                     kh * kernel_w + kw];
                    
                    thread_sum += x_val * w_val;
                }

                // Reduce partial sums within warp
                val += warp_reduce_sum(thread_sum);
            }
        }

        // First thread in warp adds bias and writes result
        if (lane_id == 0) {
            if (bias != nullptr) {
                val += bias[oc];
            }
            output[idx] = val;
        }
    }
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

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
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
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution with warp primitives (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}