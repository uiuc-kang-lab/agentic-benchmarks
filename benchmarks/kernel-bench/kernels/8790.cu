#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define warp size
constexpr int WARP_SIZE = 32;

// Kernel: Each block computes one output element using intra-block reduction
__global__ void conv_transpose1d_kernel_shared_reduction(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Each block is responsible for one output element
    // Compute output indices from blockIdx.x
    int index = blockIdx.x;
    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n = index / (L_out * C_out);

    // Total number of contributions to this output element
    int total_iters = C_in * K_w;

    // Each thread computes a partial sum over a chunk of the reduction dimension
    float partial_sum = 0.0f;
    for (int i = threadIdx.x; i < total_iters; i += blockDim.x) {
        int c_in = i / K_w;
        int k = i % K_w;
        int l_in_nom = l_out + padding - k * dilation;
        if ((l_in_nom % stride) == 0) {
            int l_in = l_in_nom / stride;
            if (l_in >= 0 && l_in < L_in) {
                int x_index = n * C_in * L_in + c_in * L_in + l_in;
                int weight_index = c_in * C_out * K_w + c_out * K_w + k;
                partial_sum += x[x_index] * weight[weight_index];
            }
        }
    }

    // Intra-warp reduction using warp-level primitives
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Use shared memory to combine results from different warps in the block
    __shared__ float warp_sums[32]; // Enough to hold sums from all warps in a block
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();

    // Let the first warp reduce the warp_sums
    float block_sum = 0.0f;
    int numWarps = blockDim.x / WARP_SIZE;
    if (threadIdx.x < numWarps) {
        block_sum = warp_sums[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        // Thread 0 writes the final result for this output element
        if (threadIdx.x == 0) {
            float out = (bias ? bias[c_out] : 0.0f) + block_sum;
            y[index] = out;
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

    torch::Tensor x = x_obj.cast<torch::Tensor>();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>();
    x = x.contiguous();
    weight = weight.contiguous();
    
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");
    
    const float* bias_ptr = nullptr;
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

    // Each block computes one output element. Total blocks = N * C_out * L_out.
    int total_elements = N * C_out * L_out;
    int threads_per_block = 256;  

    conv_transpose1d_kernel_shared_reduction<<<total_elements, threads_per_block>>>(
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
        "Conv Transpose1D forward (CUDA) with shared memory reduction",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1
    );
}
