#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
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
    extern __shared__ char shared_mem[];
    scalar_t* shared_weight = (scalar_t*)shared_mem;
    
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int out_ch_id = blockIdx.y;
    const int out_h_id = (blockIdx.z * blockDim.x + threadIdx.x) / out_width;
    const int out_w_id = (blockIdx.z * blockDim.x + threadIdx.x) % out_width;
    
    if (batch_id >= batch_size || out_ch_id >= out_channels || 
        out_h_id >= out_height || out_w_id >= out_width) return;
    
    const int out_channels_per_group = out_channels / groups;
    const int g = out_ch_id / out_channels_per_group;
    const int oc_group = out_ch_id % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;
    
    // Load weights into shared memory
    const int weights_per_thread = (kernel_h * kernel_w + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread; i++) {
        const int idx = tid * weights_per_thread + i;
        if (idx < kernel_h * kernel_w) {
            const int kh = idx / kernel_w;
            const int kw = idx % kernel_w;
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                shared_weight[ic * kernel_h * kernel_w + idx] = weight[
                    (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                    oc_group * kernel_h * kernel_w +
                    idx
                ];
            }
        }
    }
    __syncthreads();
    
    scalar_t val = (bias != nullptr) ? bias[out_ch_id] : static_cast<scalar_t>(0);
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = (out_h_id - kh * dilation + padding) / stride;
            if ((out_h_id - kh * dilation + padding) % stride != 0) continue;
            if (h_in < 0 || h_in >= in_height) continue;
            
            const int w_in = (out_w_id - kw * dilation + padding) / stride;
            if ((out_w_id - kw * dilation + padding) % stride != 0) continue;
            if (w_in < 0 || w_in >= in_width) continue;
            
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                const scalar_t x_val = input[batch_id * in_channels * in_height * in_width +
                                            (ic_start + ic) * in_height * in_width +
                                            h_in * in_width +
                                            w_in];
                val += x_val * shared_weight[ic * kernel_h * kernel_w + kh * kernel_w + kw];
            }
        }
    }
    
    const int out_idx = batch_id * out_channels * out_height * out_width +
                        out_ch_id * out_height * out_width +
                        out_h_id * out_width +
                        out_w_id;
    output[out_idx] = val;
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
    const int shared_mem_size = (in_channels / groups) * kernel_h * kernel_w * sizeof(float);
    
    dim3 blocks(
        batch_size,
        out_channels,
        (out_height * out_width + threads - 1) / threads
    );
    
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