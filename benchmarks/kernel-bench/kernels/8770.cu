#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that computes one output element per block.
// It uses shared memory reduction and warp-level primitives (__shfl_down_sync) to speed up the reduction over the (C_in * K_w) domain.

__global__ void conv_transpose1d_kernel_shared(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Each block computes one output element.
    int out_idx = blockIdx.x;  // Range: [0, N * C_out * L_out)
    int l_out = out_idx % L_out;
    int c_out = (out_idx / L_out) % C_out;
    int n = out_idx / (L_out * C_out);

    // Total reduction iterations over input channels and kernel positions.
    int R = C_in * K_w;
    float partial = 0.0f;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Each thread sums over a subset of the reduction index.
    for (int r = tid; r < R; r += blockSize) {
        int c_in = r / K_w;
        int k = r % K_w;
        int l_in_nom = l_out + padding - k * dilation;
        if (l_in_nom % stride == 0) {
            int l_in = l_in_nom / stride;
            if (l_in >= 0 && l_in < L_in) {
                float x_val = x[n * C_in * L_in + c_in * L_in + l_in];
                float w_val = weight[c_in * C_out * K_w + c_out * K_w + k];
                partial += x_val * w_val;
            }
        }
    }

    // Allocate shared memory for reduction
    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction using __shfl_down_sync for the final stage
    float sum = sdata[tid];
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            y[out_idx] = sum;
        }
    }
}


torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {

    // Convert py::object to torch::Tensor and ensure contiguity
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    // Dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    int output_elements = N * C_out * L_out;
    int threads = 256; // Number of threads per block
    dim3 blocks(output_elements);

    // Allocate shared memory: one float per thread
    size_t shared_mem_size = threads * sizeof(float);

    conv_transpose1d_kernel_shared<<<blocks, threads, shared_mem_size>>>(
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
        "Conv Transpose1D forward with shared memory reduction (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1
    );
}
