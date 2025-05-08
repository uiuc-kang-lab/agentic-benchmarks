#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

__global__ void conv1d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    const int N,
    const int C_in,
    const int L_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out
)
{
    // Each warp processes multiple output elements
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 0x1F;  // Lane ID within warp (0-31)
    const unsigned int wid = tid >> 5;        // Warp ID within block
    
    // Global position
    const unsigned int gid = blockIdx.x * (blockDim.x >> 5) + wid;
    const unsigned int total_warps = gridDim.x * (blockDim.x >> 5);
    
    // Process multiple elements per warp
    for (int idx = gid; idx < N * C_out * L_out; idx += total_warps) {
        const int out_pos = idx % L_out;
        const int out_ch = (idx / L_out) % C_out;
        const int n = idx / (L_out * C_out);
        
        // Determine group index
        const int group_size_out = C_out / groups;
        const int group_size_in = C_in / groups;
        const int group_idx = out_ch / group_size_out;
        
        float val = 0.0f;
        
        // Distribute channel and kernel offset calculations across warp lanes
        for (int local_in_ch = lane_id; local_in_ch < group_size_in; local_in_ch += 32) {
            const int in_ch = group_idx * group_size_in + local_in_ch;
            
            for (int k = 0; k < K; k++) {
                const int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    const float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    const float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
                    val += x_val * w_val;
                }
            }
        }
        
        // Warp reduction using shuffle operations
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        // First lane writes the result
        if (lane_id == 0) {
            if (bias_ptr) {
                val += bias_ptr[out_ch];
            }
            y[n * (C_out * L_out) + out_ch * L_out + out_pos] = val;
        }
    }
}

at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int64_t N = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K = w_sizes[2];

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    const int blockSize = 256;
    const int numWarps = blockSize / 32;
    const int numBlocks = (N * C_out * L_out + numWarps - 1) / numWarps;

    conv1d_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel failed: ", cudaGetErrorString(err));

    return y;
}

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
            return conv1d_forward_impl(x, weight, bias, stride, padding, dilation, groups);
        },
        "Warp-shuffle optimized 1D Convolution forward (CUDA) with optional bias"
    );
}