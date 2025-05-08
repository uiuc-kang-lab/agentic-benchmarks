#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* output,
    const int N, const int C, const int D, const int H, const int W,
    const int K, const int T, const int R, const int S,
    const int out_D, const int out_H, const int out_W,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = tid / (out_D * out_H * out_W);
    const int spatial_idx = tid % (out_D * out_H * out_W);
    
    if (batch_idx >= N) return;
    
    const int out_d = (spatial_idx / (out_H * out_W));
    const int out_h = (spatial_idx % (out_H * out_W)) / out_W;
    const int out_w = spatial_idx % out_W;
    
    for (int k = 0; k < K; k++) {
        scalar_t sum = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T; t++) {
                const int in_d = (out_d + pad_d - t) / stride_d;
                if (in_d < 0 || in_d >= D) continue;
                
                for (int r = 0; r < R; r++) {
                    const int in_h = (out_h + pad_h - r) / stride_h;
                    if (in_h < 0 || in_h >= H) continue;
                    
                    for (int s = 0; s < S; s++) {
                        const int in_w = (out_w + pad_w - s) / stride_w;
                        if (in_w < 0 || in_w >= W) continue;
                        
                        const scalar_t input_val = __ldg(&input[
                            ((batch_idx * C + c) * D + in_d) * H * W +
                            in_h * W + in_w]);
                        
                        const scalar_t weight_val = __ldg(&weight[
                            ((k * C + c) * T + t) * R * S +
                            r * S + s]);
                        
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        
        output[((batch_idx * K + k) * out_D + out_d) * out_H * out_W +
               out_h * out_W + out_w] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_depth = x.size(2);
    const auto in_height = x.size(3);
    const auto in_width = x.size(4);
    
    const auto out_channels = weight.size(1) * groups;
    const auto kernel_depth = weight.size(2);
    const auto kernel_height = weight.size(3);
    const auto kernel_width = weight.size(4);
    
    const auto out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] +
                          kernel_depth + output_padding[0];
    const auto out_height = (in_height - 1) * stride[1] - 2 * padding[1] +
                           kernel_height + output_padding[1];
    const auto out_width = (in_width - 1) * stride[2] - 2 * padding[2] +
                          kernel_width + output_padding[2];
    
    auto output = torch::zeros({batch_size, out_channels, out_depth,
                               out_height, out_width}, x.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_depth * out_height * out_width +
                       threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv_transpose3d_kernel", ([&] {
        conv_transpose3d_kernel<<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, in_depth, in_height, in_width,
            out_channels, kernel_depth, kernel_height, kernel_width,
            out_depth, out_height, out_width,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2]
        );
    }));
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, out_channels, 1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}