#include <torch/extension.h>
#include <cuda_runtime.h>

#define ALIGN_BYTES 16

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    auto input = x.contiguous();
    auto kernel = weight.contiguous();
    
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_h = input.size(2);
    int64_t in_w = input.size(3);
    
    int64_t out_channels = kernel.size(1);
    int64_t k_h = kernel.size(2);
    int64_t k_w = kernel.size(3);
    
    int64_t s_h = stride[0], s_w = stride[1];
    int64_t p_h = padding[0], p_w = padding[1];
    
    int64_t out_h = (in_h - 1) * s_h + k_h - 2 * p_h;
    int64_t out_w = (in_w - 1) * s_w + k_w - 2 * p_w;
    
    auto output = torch::empty({
        batch_size,
        out_channels,
        out_h,
        out_w
    }, input.options());
    
    auto input_a = input.accessor<float,4>();
    auto kernel_a = kernel.accessor<float,4>();
    auto output_a = output.accessor<float,4>();
    
    int threads = 256;
    dim3 blocks(
        (out_w + 15) / 16,
        (out_h + 15) / 16,
        batch_size
    );
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv_transpose2d", [&] {
        auto input_ptr = input.data_ptr<scalar_t>();
        auto kernel_ptr = kernel.data_ptr<scalar_t>();
        auto output_ptr = output.data_ptr<scalar_t>();
        
        if (!bias_obj.is_none()) {
            auto bias = bias_obj.cast<torch::Tensor>();
            auto bias_a = bias.accessor<scalar_t,1>();
            __global__ void conv_transpose2d_kernel_bias(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ kernel,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k_h,
    const int k_w,
    const int s_h,
    const int s_w,
    const int p_h,
    const int p_w
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t sum = bias[oc];
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    // Calculate input position
                    const int in_y = (y + p_h - kh) / s_h;
                    const int in_x = (x + p_w - kw) / s_w;
                    
                    // Check if the input position is valid and aligned with stride
                    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w &&
                        (y + p_h - kh) % s_h == 0 && (x + p_w - kw) % s_w == 0) {
                        
                        const scalar_t in_val = __ldg(&input[
                            ((b * in_channels + ic) * in_h + in_y) * in_w + in_x
                        ]);
                        
                        const scalar_t k_val = __ldg(&kernel[
                            ((oc * in_channels + ic) * k_h + kh) * k_w + kw
                        ]);
                        
                        sum += in_val * k_val;
                    }
                }
            }
        }
        
        output[((b * out_channels + oc) * out_h + y) * out_w + x] = sum;
    }
}

conv_transpose2d_kernel_bias<<<blocks, threads>>>(
    input_ptr,
    kernel_ptr,
    bias_a.data(),
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    k_h,
    k_w,
    s_h,
    s_w,
    p_h,
    p_w
);
        } else {
            __global__ void conv_transpose2d_kernel_bias(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ kernel,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k_h,
    const int k_w,
    const int s_h,
    const int s_w,
    const int p_h,
    const int p_w
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t sum = bias[oc];
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    // Calculate input position
                    const int in_y = (y + p_h - kh) / s_h;
                    const int in_x = (x + p_w - kw) / s_w;
                    
                    // Check if the input position is valid and aligned with stride
                    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w &&
                        (y + p_h - kh) % s_h == 0 && (x + p_w - kw) % s_w == 0) {
                        
                        const scalar_t in_val = __ldg(&input[
                            ((b * in_channels + ic) * in_h + in_y) * in_w + in_x
                        ]);
                        
                        const scalar_t k_val = __ldg(&kernel[
                            ((oc * in_channels + ic) * k_h + kh) * k_w + kw
                        ]);
                        
                        sum += in_val * k_val;
                    }
                }
            }
        }
        
        output[((b * out_channels + oc) * out_h + y) * out_w + x] = sum;
    }
}

conv_transpose2d_kernel_bias<<<blocks, threads>>>(
    input_ptr,
    kernel_ptr,
    bias_a.data(),
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    k_h,
    k_w,
    s_h,
    s_w,
    p_h,
    p_w
);
        }
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
