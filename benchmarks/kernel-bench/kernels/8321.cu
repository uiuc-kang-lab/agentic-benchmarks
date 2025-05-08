#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// -----------------------------------------------------------------
// Optimized 1D convolution CUDA kernel with shared memory reduction
// -----------------------------------------------------------------
__global__ void conv1d_forward_kernel_shared_reduction(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // can be null if no bias
    float* __restrict__ y,
    const int N,         // batch size
    const int C_in,      // input channels
    const int L_in,      // input length
    const int C_out,     // output channels
    const int K,         // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out      // output length
) {
    const int warpSize = 32;
    int n = blockIdx.x;
    int out_ch = blockIdx.y;
    int thread_out = threadIdx.x;

    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_idx = out_ch / group_size_out;

    extern __shared__ float shmem[];
    float* shw = shmem;
    float* shx = shmem + group_size_in * K;

    int total_w_elements = group_size_in * K;
    for (int i = threadIdx.x; i < total_w_elements; i += blockDim.x) {
        shw[i] = w[out_ch * total_w_elements + i];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int out_pos = thread_out; out_pos < L_out; out_pos += blockDim.x) {
        for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
            int in_ch = group_idx * group_size_in + local_in_ch;
            for (int k = 0; k < K; k++) {
                int in_pos = out_pos * stride + k * dilation - padding;
                int clamped_in_pos = in_pos < 0 ? 0 : (in_pos >= L_in ? (L_in - 1) : in_pos);
                float mask = ((unsigned)in_pos < (unsigned)L_in) ? 1.0f : 0.0f;
                int x_index = n * (C_in * L_in) + in_ch * L_in + clamped_in_pos;
                int w_index = local_in_ch * K + k;
                sum += mask * x[x_index] * shw[w_index];
            }
        }
        shx[threadIdx.x] = sum;
        __syncthreads();

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (threadIdx.x % warpSize == 0) {
            atomicAdd(&y[n * (C_out * L_out) + out_ch * L_out + out_pos], sum);
        }
    }

    if (bias_ptr && thread_out < L_out) {
        y[n * (C_out * L_out) + out_ch * L_out + thread_out] += bias_ptr[out_ch];
    }
}

// -------------------------------------------------------
// Implementation of conv1d forward with shared memory reduction
// -------------------------------------------------------
at::Tensor conv1d_forward_impl_shared_reduction(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    dim3 grid(N, C_out);
    int threads = L_out < 256 ? L_out : 256;
    int sharedMem = (C_in / groups) * K * sizeof(float) + threads * sizeof(float);

    conv1d_forward_kernel_shared_reduction<<<grid, threads, sharedMem>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_shared_reduction failed: ", cudaGetErrorString(err));

    return y;
}

// -----------------------------------------------------
// Pybind11 binding for the optimized convolution kernel
// -----------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor weight,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl_shared_reduction(x, weight, bias, stride, padding, dilation, groups);
        },
        "Optimized 1D Convolution forward (CUDA) with shared memory reduction"
    );
}