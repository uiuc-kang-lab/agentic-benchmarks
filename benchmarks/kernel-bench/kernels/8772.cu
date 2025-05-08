#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel_coalesced(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    extern __shared__ float shared_weight[];
    
    // Block handles one batch, one output channel at a time
    const int n = blockIdx.y;
    const int c_out = blockIdx.z;
    
    // Each thread processes consecutive l_out positions
    const int tid = threadIdx.x;
    const int l_out_start = blockIdx.x * blockDim.x + tid;
    
    // Load weights into shared memory
    if (tid < K_w) {
        for (int c_in = 0; c_in < C_in; c_in++) {
            shared_weight[c_in * K_w + tid] = weight[c_in * C_out * K_w + c_out * K_w + tid];
        }
    }
    __syncthreads();

    // Initialize output value with bias if present
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Process only if within valid output range
    if (l_out_start < L_out) {
        value = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        // Process input channels
        for (int c_in = 0; c_in < C_in; c_in++) {
            const int x_base = n * C_in * L_in + c_in * L_in;
            
            // Process kernel positions
            for (int k_w = 0; k_w < K_w; k_w++) {
                int l_in_nom = l_out_start + padding - k_w * dilation;
                int l_in = l_in_nom / stride;
                
                if ((l_in_nom % stride == 0) && (l_in >= 0) && (l_in < L_in)) {
                    float x_val = x[x_base + l_in];
                    float w_val = shared_weight[c_in * K_w + k_w];
                    value += x_val * w_val;
                }
            }
        }
        
        // Write result to global memory
        y[n * C_out * L_out + c_out * L_out + l_out_start] = value;
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

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Configure kernel launch parameters
    const int THREADS_PER_BLOCK = 128;  // Multiple of warp size (32)
    const int l_out_blocks = (L_out + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 blocks(l_out_blocks, N, C_out);
    dim3 threads(THREADS_PER_BLOCK);
    
    // Shared memory size for weights
    int shared_mem_size = C_in * K_w * sizeof(float);

    conv_transpose1d_kernel_coalesced<<<blocks, threads, shared_mem_size>>>(
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
        "Coalesced Memory Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}