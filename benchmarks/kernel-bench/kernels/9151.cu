#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;


// CUDA kernel for ConvTranspose2d with reduced warp divergence
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,       // Input tensor: [N, in_channels, inH, inW]
    const float* __restrict__ weight,  // Weight tensor: [in_channels, out_channels, kernelH, kernelW]
    const float* __restrict__ bias,    // Bias tensor: [out_channels] (can be nullptr)
    float* __restrict__ output,        // Output tensor: [N, out_channels, outH, outW]
    int N,
    int in_channels,
    int out_channels,
    int inH,
    int inW,
    int outH,
    int outW,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * outH * outW;
    if (idx < total) {
        // Map linear thread index to output indices
        int ow = idx % outW;
        int tmp = idx / outW;
        int oh = tmp % outH;
        int tmp2 = tmp / outH;
        int oc = tmp2 % out_channels;
        int n = tmp2 / out_channels;

        float value = 0.0f;
        
        // Loop over input channels
        for (int ic = 0; ic < in_channels; ic++) {
            // Compute effective coordinate for height
            int t_h = oh + padH;
            // Compute starting kernel index for height that satisfies: (t_h - k) mod strideH == 0
            int kH_start = t_h % strideH;  
            // Compute lower bound to ensure input index is in range: (t_h - k)/strideH < inH
            int min_kH = t_h - (inH - 1) * strideH;
            // Compute adjustment in a branchless manner using a ternary operator
            int add_kH = (min_kH > kH_start) ? ((min_kH - kH_start + strideH - 1) / strideH) : 0;
            int kH_valid = kH_start + add_kH * strideH;
            // Upper bound: kH must be <= t_h (i.e. t_h + 1) and within kernelH
            int kH_end = (t_h + 1 < kernelH) ? t_h + 1 : kernelH;
            
            // Loop over valid kernel height indices
            for (int kH = kH_valid; kH < kH_end; kH += strideH) {
                int in_h = (t_h - kH) / strideH;  // Guaranteed to be integer and in range by construction

                // Compute effective coordinate for width
                int t_w = ow + padW;
                int kW_start = t_w % strideW;
                int min_kW = t_w - (inW - 1) * strideW;
                int add_kW = (min_kW > kW_start) ? ((min_kW - kW_start + strideW - 1) / strideW) : 0;
                int kW_valid = kW_start + add_kW * strideW;
                int kW_end = (t_w + 1 < kernelW) ? t_w + 1 : kernelW;
                
                // Loop over valid kernel width indices
                for (int kW = kW_valid; kW < kW_end; kW += strideW) {
                    int in_w = (t_w - kW) / strideW;
                    
                    // Calculate linear indices
                    int x_index = ((n * in_channels + ic) * inH + in_h) * inW + in_w;
                    int weight_index = ((ic * out_channels + oc) * kernelH + kH) * kernelW + kW;
                    
                    value += x[x_index] * weight[weight_index];
                }
            }
        }
        // Add bias if provided (pointer is uniform across threads)
        if (bias != nullptr) {
            value += bias[oc];
        }

        int out_index = ((n * out_channels + oc) * outH + oh) * outW + ow;
        output[out_index] = value;
    }
}

// Host launcher for the CUDA kernel

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Extract dimensions
    const int N = x.size(0);
    const int in_channels = x.size(1);
    const int inH = x.size(2);
    const int inW = x.size(3);
    
    const int kernelH = weight.size(2);
    const int kernelW = weight.size(3);
    const int out_channels = weight.size(1);
    
    // Compute output dimensions for transposed convolution
    const int outH = (inH - 1) * stride[0] - 2 * padding[0] + kernelH;
    const int outW = (inW - 1) * stride[1] - 2 * padding[1] + kernelW;
    
    auto output = torch::zeros({N, out_channels, outH, outW}, x.options());
    
    const int total = N * out_channels * outH * outW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, in_channels, out_channels,
        inH, inW, outH, outW,
        kernelH, kernelW,
        stride[0], stride[1],
        padding[0], padding[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_cuda, "Fast Conv Transpose2D forward with minimized warp divergence",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("stride"),
          py::arg("padding"));
}
