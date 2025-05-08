#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a 3D grid mapping to directly map the output dimensions
// (batch, output channel, output position) to the grid dimensions. This allows
// for a more intuitive and efficient distribution of threads across the work domain.

__global__ void conv_transpose1d_kernel_3d(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Map grid dimensions: 
    //   blockIdx.x -> batch index (n)
    //   blockIdx.y -> output channel (c_out)
    //   blockIdx.z and threadIdx.x -> output position (l_out)
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int l_out = blockIdx.z * blockDim.x + threadIdx.x; if (l_out >= L_out) return;

    if (l_out >= L_out) return;

    // Initialize with bias if present
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    int x_batch_offset = n * C_in * L_in;

    // For each input channel, accumulate contributions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_channel_offset = x_batch_offset + c_in * L_in;
        int w_offset = c_in * C_out * K_w + c_out * K_w;
        
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            // Check if this output position corresponds to a valid input index
            if (l_in_nom % stride == 0) {
                int l_in = l_in_nom / stride;
                if (l_in >= 0 && l_in < L_in) {
                    value += x[x_channel_offset + l_in] * weight[w_offset + k_w];
                }
            }
        }
    }
    
    // Write the computed value to the output tensor
    int y_index = n * (C_out * L_out) + c_out * L_out + l_out;
    y[y_index] = value;
}


// Host function: sets up tensor checks and kernel launch parameters
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,            // x: torch.Tensor
    py::object weight_obj,       // weight: torch.Tensor
    py::object bias_obj = py::none(),  // bias: torch.Tensor or None
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);

    // Compute the length of the output
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Configure the kernel launch parameters using a 3D grid
    // Grid dimensions: 
    //   x-dimension: batch (N)
    //   y-dimension: output channels (C_out)
    //   z-dimension: divides the output length (L_out) among thread blocks
    int threads = 256;  // Number of threads per block in x dimension
    dim3 block(threads);
    dim3 grid(N, C_out, (L_out + threads - 1) / threads);

    conv_transpose1d_kernel_3d<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Conv Transpose1D forward (CUDA) with 3D grid indexing",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
