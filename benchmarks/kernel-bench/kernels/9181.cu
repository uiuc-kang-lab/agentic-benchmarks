#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Define maximum constant memory size in number of floats (e.g., 16384 floats ~64KB)
#define MAX_CONST_SIZE 16384

// Declare constant memory for the weight tensor
__constant__ float const_weight[MAX_CONST_SIZE];

// CUDA kernel for 2D transposed convolution using constant memory for the weight
// Input tensor dimensions: [batch, in_channels, in_h, in_w]
// Weight tensor dimensions (assumed): [in_channels, out_channels, kH, kW]
// Output dimensions are computed based on stride and padding
__global__ void conv_transpose2d_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const float* __restrict__ bias,
                                          int batch, int in_channels, int out_channels,
                                          int in_h, int in_w, int out_h, int out_w,
                                          int kH, int kW, int stride_h, int stride_w,
                                          int pad_h, int pad_w) {
    // Compute output spatial indices
    int out_w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // blockIdx.z encodes both batch index and output channel
    int n_oc = blockIdx.z; 
    int batch_idx = n_oc / out_channels;
    int oc = n_oc % out_channels;

    if (out_w_idx < out_w && out_h_idx < out_h) {
        float total = 0.0f;
        // For each input channel, iterate over kernel elements
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kH; kh++) {
                // Calculate candidate input row index
                int in_h_idx_candidate = out_h_idx + pad_h - kh;
                if (in_h_idx_candidate % stride_h != 0) continue;
                in_h_idx_candidate /= stride_h;
                if (in_h_idx_candidate < 0 || in_h_idx_candidate >= in_h) continue;
                
                for (int kw = 0; kw < kW; kw++) {
                    // Calculate candidate input column index
                    int in_w_idx_candidate = out_w_idx + pad_w - kw;
                    if (in_w_idx_candidate % stride_w != 0) continue;
                    in_w_idx_candidate /= stride_w;
                    if (in_w_idx_candidate < 0 || in_w_idx_candidate >= in_w) continue;

                    // Compute the linear index for the input tensor
                    int input_index = ((batch_idx * in_channels + ic) * in_h + in_h_idx_candidate) * in_w + in_w_idx_candidate;
                    // Compute the linear index for the weight from constant memory
                    int weight_index = (((ic * out_channels + oc) * kH) + kh) * kW + kw;
                    total += input[input_index] * const_weight[weight_index];
                }
            }
        }
        // Add bias if provided
        if (bias != nullptr) {
            total += bias[oc];
        }
        // Compute the output tensor linear index and store the result
        int out_index = ((batch_idx * out_channels + oc) * out_h + out_h_idx) * out_w + out_w_idx;
        output[out_index] = total;
    }
}

// Host function: copies weight tensor into constant memory and launches the kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Ensure input and weight are contiguous
    auto input = x.contiguous();
    auto weight_cont = weight.contiguous();

    // Retrieve dimensions
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    // Expected weight dimensions: [in_channels, out_channels, kH, kW]
    int kH = weight_cont.size(2);
    int kW = weight_cont.size(3);
    int out_channels = weight_cont.size(1);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];

    // Compute output spatial dimensions
    int out_h = (in_h - 1) * stride_h + kH - 2 * pad_h;
    int out_w = (in_w - 1) * stride_w + kW - 2 * pad_w;

    auto output = torch::zeros({batch, out_channels, out_h, out_w}, input.options());

    // Copy weight into constant memory if it fits
    int weight_numel = weight_cont.numel();
    if (weight_numel > MAX_CONST_SIZE) {\n    // Allocate global memory for the weight tensor\n    float* global_weight;\n    cudaMalloc(&global_weight, weight_numel * sizeof(float));\n    cudaMemcpy(global_weight, weight_cont.data_ptr<float>(), weight_numel * sizeof(float), cudaMemcpyHostToDevice);\n    // Launch kernel with global memory\n    conv_transpose2d_kernel<<<grid, block>>>(\n        input.data_ptr<float>(),\n        output.data_ptr<float>(),\n        bias_ptr,\n        batch, in_channels, out_channels,\n        in_h, in_w, out_h, out_w,\n        kH, kW,\n        stride_h, stride_w,\n        pad_h, pad_w,\n        global_weight\n    );\n    cudaFree(global_weight);\n    return output;\n} else {\n    cudaMemcpyToSymbol(const_weight, weight_cont.data_ptr<float>(), weight_numel * sizeof(float));\n}
    cudaMemcpyToSymbol(const_weight, weight_cont.data_ptr<float>(), weight_numel * sizeof(float));

    // Prepare bias pointer if bias is provided
    float* bias_ptr = nullptr;
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>().contiguous();
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Configure CUDA kernel launch dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch * out_channels);

    conv_transpose2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        batch, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with constant memory for weights",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
