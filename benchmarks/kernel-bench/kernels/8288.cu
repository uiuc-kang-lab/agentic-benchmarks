#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

__device__ __forceinline__ float4 ldg4(const float* ptr) {
    float4 v;
    v.x = __ldg(ptr);
    v.y = __ldg(ptr + 1);
    v.z = __ldg(ptr + 2);
    v.w = __ldg(ptr + 3);
    return v;
}

__global__ void conv1d_forward_kernel_optimized(
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
) {
    const int tid = threadIdx.x;
    const int out_ch = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int base_out_pos = (blockIdx.y * blockDim.x + tid) * 4;

    if (base_out_pos >= L_out) return;

    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;
    
    // Pre-compute input batch offset
    const int batch_offset = batch_idx * (C_in * L_in);
    
    // Initialize output accumulators
    float4 out_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Pre-compute weight offset for current output channel
    const int weight_offset = out_ch * (group_size_in * K);
    
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        const int in_ch = group_idx * group_size_in + local_in_ch;
        const int in_ch_offset = batch_offset + in_ch * L_in;
        
        // Prefetch first weight
        float w_curr = __ldg(&w[weight_offset + local_in_ch * K]);
        
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            // Prefetch next weight for next iteration
            float w_next = (k < K-1) ? __ldg(&w[weight_offset + local_in_ch * K + k + 1]) : 0.0f;
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int out_pos = base_out_pos + i;
                if (out_pos < L_out) {
                    const int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        float x_val = __ldg(&x[in_ch_offset + in_pos]);
                        float* acc = reinterpret_cast<float*>(&out_val);
                        acc[i] += x_val * w_curr;
                    }
                }
            }
            w_curr = w_next;
        }
    }

    // Add bias if present using vectorized load
    if (bias_ptr) {
        const float bias = __ldg(&bias_ptr[out_ch]);
        float* acc = reinterpret_cast<float*>(&out_val);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += bias;
        }
    }

    // Calculate output offset
    const int out_offset = batch_idx * (C_out * L_out) + out_ch * L_out + base_out_pos;
    const int remaining = min(4, L_out - base_out_pos);

    // Store results using vectorized store when possible
    if (remaining == 4 && ((reinterpret_cast<uintptr_t>(&y[out_offset]) & 15) == 0)) {
        *reinterpret_cast<float4*>(&y[out_offset]) = out_val;
    } else {
        const float* acc = reinterpret_cast<float*>(&out_val);
        #pragma unroll
        for (int i = 0; i < remaining; ++i) {
            y[out_offset + i] = acc[i];
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
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int C_out = w_sizes[0];
    int K = w_sizes[2];

    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    const int threads = 256;
    dim3 blocks(
        C_out,
        (L_out + threads * 4 - 1) / (threads * 4),
        N
    );

    conv1d_forward_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_optimized failed: ", cudaGetErrorString(err));

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
        "Optimized 1D Convolution forward (CUDA)"
    );
}