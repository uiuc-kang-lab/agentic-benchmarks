#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel_strided(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const int N, const int C_in, const int C_out, 
    const int L_in, const int L_out, const int K_w,
    const int stride, const int padding, const int dilation) {

    // Grid-stride loop pattern
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_step = blockDim.x * gridDim.x;
    const int total_elements = N * C_out * L_out;
    
    // Pre-compute constant offsets
    const int L_in_stride = L_in;
    const int L_out_stride = L_out;
    const int weight_stride = K_w;
    
    // Process multiple elements per thread using stride loop
    for (int idx = tid; idx < total_elements; idx += stride_step) {
        // Calculate 3D position from linear index
        const int l_out = idx % L_out;
        const int c_out = (idx / L_out) % C_out;
        const int n = idx / (L_out * C_out);
        
        // Initialize output value
        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        // Pre-compute base offsets for input and weight access
        const int x_batch_offset = n * C_in * L_in_stride;
        const int weight_out_offset = c_out * weight_stride;
        
        // Main computation loop with better memory access pattern
        #pragma unroll 4
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int x_offset = x_batch_offset + c_in * L_in_stride;
            const int w_offset = c_in * C_out * weight_stride + weight_out_offset;
            
            #pragma unroll 2
            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int l_in_nom = l_out + padding - k_w * dilation;
                if (l_in_nom % stride == 0) {
                    const int l_in = l_in_nom / stride;
                    if (l_in >= 0 && l_in < L_in) {
                        sum += x[x_offset + l_in] * weight[w_offset + k_w];
                    }
                }
            }
        }
        
        // Write result to global memory
        y[n * C_out * L_out_stride + c_out * L_out_stride + l_out] = sum;
    }
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
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

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int L_in = x.size(2);
    const int K_w = weight.size(2);
    const int C_out = weight.size(1);
    const int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Optimized launch parameters
    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    const int total_elements = N * C_out * L_out;
    const int num_blocks = std::min(
        65535,
        (total_elements + threads_per_block * elements_per_thread - 1) / 
        (threads_per_block * elements_per_thread)
    );

    conv_transpose1d_kernel_strided<<<num_blocks, threads_per_block>>>(
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
        "Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}