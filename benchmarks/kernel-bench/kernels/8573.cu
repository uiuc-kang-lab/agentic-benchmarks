#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace torch::indexing;

template<typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int groups) {
    
    extern __shared__ scalar_t s_weights[];
    
    const int n = blockIdx.z;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    const int c_out = blockIdx.z % C_out;
    const int weight_size = C_in * kH * kW;
    
    // Cooperative weight loading with single sync
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < weight_size; i += blockDim.x * blockDim.y) {
        s_weights[i] = weight[c_out * weight_size + i];
    }
    __syncthreads();
    
    scalar_t sum = 0;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int h_in = (h_out - kh + padding_h) / stride_h;
                const int w_in = (w_out - kw + padding_w) / stride_w;
                
                if ((h_out - kh + padding_h) % stride_h != 0) continue;
                if ((w_out - kw + padding_w) % stride_w != 0) continue;
                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;
                
                const scalar_t x_val = x[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in];
                const int weight_idx = c_in * kH * kW + kh * kW + kw;
                sum += x_val * s_weights[weight_idx];
            }
        }
    }
    
    if (bias) sum += bias[c_out];
    
    output[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride,
    py::object padding,
    py::object output_padding,
    int64_t groups) {
    
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    
    const int C_out = weight.size(1) * groups;
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    
    const int H_out = (H_in - 1) * stride_vec[0] - 2 * padding_vec[0] + kH;
    const int W_out = (W_in - 1) * stride_vec[1] - 2 * padding_vec[1] + kW;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    const int threads = 16;
    const dim3 blocks(
        (W_out + threads - 1) / threads,
        (H_out + threads - 1) / threads,
        N * C_out
    );
    
    const size_t shared_mem = C_in * kH * kW * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d", [&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, dim3(threads, threads), shared_mem>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            C_in, C_out,
            H_in, W_in,
            H_out, W_out,
            kH, kW,
            stride_vec[0], stride_vec[1],
            padding_vec[0], padding_vec[1],
            groups
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}