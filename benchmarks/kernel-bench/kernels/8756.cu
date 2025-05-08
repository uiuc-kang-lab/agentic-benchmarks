#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv_transpose1d_kernel_warp_opt(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    const int total_elements = N * C_out * L_out;
    const int elements_per_warp = (total_elements + gridDim.x * warps_per_block - 1) / 
                                (gridDim.x * warps_per_block);

    const int warp_start_idx = global_warp_id * elements_per_warp;
    
    for (int local_idx = 0; local_idx < elements_per_warp; local_idx++) {
        const int idx = warp_start_idx + local_idx;
        if (idx >= total_elements) break;

        const int l_out = idx % L_out;
        const int c_out = (idx / L_out) % C_out;
        const int n = idx / (L_out * C_out);

        float partial_sum = 0.0f;
        
        for (int c_in = lane_id; c_in < C_in; c_in += warp_size) {
            const int x_base = n * C_in * L_in + c_in * L_in;
            const int w_base = c_in * C_out * K_w + c_out * K_w;

            for (int k_w = 0; k_w < K_w; k_w++) {
                const int l_in_nom = l_out + padding - k_w * dilation;
                const int mod_valid = (l_in_nom % stride == 0);
                const int l_in = l_in_nom / stride;
                const int range_valid = (l_in >= 0 && l_in < L_in);
                const int valid = mod_valid && range_valid;

                const float x_val = valid ? x[x_base + l_in] : 0.0f;
                const float w_val = weight[w_base + k_w];
                partial_sum += x_val * w_val;
            }
        }

        float final_sum = warp_reduce_sum(partial_sum);

        if (lane_id == 0) {
            if (bias != nullptr) {
                final_sum += bias[c_out];
            }
            y[idx] = final_sum;
        }
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

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int blocks = num_sm * 8;

    conv_transpose1d_kernel_warp_opt<<<blocks, threads_per_block>>>(
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
        "Warp-optimized Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}