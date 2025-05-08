#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ConvTranspose1D with shared memory optimization
template<int BLOCK_SIZE = 128, int CHANNELS_PER_BLOCK = 32>
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem;
    float* shared_weight = shared_mem + BLOCK_SIZE * CHANNELS_PER_BLOCK;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Calculate output position
    int output_idx = bid * BLOCK_SIZE + tid;
    if (output_idx >= N * C_out * L_out) return;
    
    int l_out = output_idx % L_out;
    int c_out = (output_idx / L_out) % C_out;
    int n = output_idx / (L_out * C_out);

    float result = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Process input channels in chunks
    for (int c_in_start = 0; c_in_start < C_in; c_in_start += CHANNELS_PER_BLOCK) {
        int channels_this_iter = min(CHANNELS_PER_BLOCK, C_in - c_in_start);

        // Collaborative loading of input data into shared memory
        for (int c_offset = 0; c_offset < channels_this_iter; c_offset++) {
            int c_in = c_in_start + c_offset;
            
            // Calculate potential input positions for this thread
            int l_in_nom = l_out + padding;
            int l_in_base = l_in_nom / stride;
            
            if (l_in_base >= 0 && l_in_base < L_in) {
                shared_x[c_offset * BLOCK_SIZE + tid] = 
                    x[n * C_in * L_in + c_in * L_in + l_in_base];
            } else {
                shared_x[c_offset * BLOCK_SIZE + tid] = 0.0f;
            }

            // Load weights
            if (tid < K_w) {
                shared_weight[c_offset * K_w + tid] = 
                    weight[c_in * C_out * K_w + c_out * K_w + tid];
            }
        }
        
        __syncthreads();

        // Compute partial results using shared memory
        for (int c_offset = 0; c_offset < channels_this_iter; c_offset++) {
            for (int k_w = 0; k_w < K_w; k_w++) {
                int l_in_nom = l_out + padding - k_w * dilation;
                if (l_in_nom % stride == 0) {
                    int l_in = l_in_nom / stride;
                    if (l_in >= 0 && l_in < L_in) {
                        float x_val = shared_x[c_offset * BLOCK_SIZE + tid];
                        float w_val = shared_weight[c_offset * K_w + k_w];
                        result += x_val * w_val;
                    }
                }
            }
        }
        
        __syncthreads();
    }

    if (output_idx < N * C_out * L_out) {
        y[output_idx] = result;
    }
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1)
{
    torch::Tensor x = x_obj.cast<torch::Tensor>();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>();

    x = x.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>();
        bias = bias.contiguous();
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

    constexpr int BLOCK_SIZE = 128;
    constexpr int CHANNELS_PER_BLOCK = 32;
    
    int total_elements = N * C_out * L_out;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Calculate shared memory size
    int shared_mem_size = (BLOCK_SIZE * CHANNELS_PER_BLOCK + CHANNELS_PER_BLOCK * K_w) * sizeof(float);

    conv_transpose1d_kernel<BLOCK_SIZE, CHANNELS_PER_BLOCK><<<blocks, BLOCK_SIZE, shared_mem_size>>>(
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