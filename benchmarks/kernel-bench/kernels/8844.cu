#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int THREADS>
__global__ void conv_transpose1d_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    extern __shared__ float smem[];
    float* x_shared = smem;
    float* w_shared = smem + C_in * K_w;

    int tid = threadIdx.x;
    int n = blockIdx.y;
    int c_out = blockIdx.z;
    int l_out = blockIdx.x * blockDim.x + threadIdx.x;

    if(l_out >= L_out) return;

    // Initialize output with bias
    float result = (bias != nullptr) ? bias[c_out] : 0.0f;

    for(int c_in_block = 0; c_in_block < C_in; c_in_block += gridDim.z) {
        int c_in = c_in_block + blockIdx.z;
        if(c_in >= C_in) continue;

        // Cooperatively load weights into shared memory
        if(tid < K_w) {
            w_shared[c_in * K_w + tid] = weight[c_in * C_out * K_w + c_out * K_w + tid];
        }

        // Load input elements needed for this channel
        for(int k = tid; k < K_w; k += blockDim.x) {
            int l_in_nom = l_out + padding - k * dilation;
            int l_in = (l_in_nom % stride == 0) ? (l_in_nom / stride) : -1;
            if(l_in >= 0 && l_in < L_in) {
                x_shared[c_in * K_w + k] = x[n * C_in * L_in + c_in * L_in + l_in];
            } else {
                x_shared[c_in * K_w + k] = 0.0f;
            }
        }

        __syncthreads();

        // Compute with shared memory
        for(int k = 0; k < K_w; ++k) {
            result += x_shared[c_in * K_w + k] * w_shared[c_in * K_w + k];
        }

        __syncthreads();
    }

    if(l_out < L_out) {
        y[n * C_out * L_out + c_out * L_out + l_out] = result;
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

    const int threads = 256;
    const int blocks_x = (L_out + threads - 1) / threads;
    dim3 blocks(blocks_x, N, C_out);

    size_t shared_mem = (C_in * K_w + C_in * K_w) * sizeof(float);

    conv_transpose1d_shared_kernel<threads><<<blocks, threads, shared_mem>>>(
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
        "Conv Transpose1D forward (Shared Memory Optimized)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}