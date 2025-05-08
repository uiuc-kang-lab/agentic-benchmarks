#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <vector>

namespace py = pybind11;

// Kernel remains similar but optimized for stream-based execution
__global__ void conv1d_forward_kernel_pipelined(
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
    const int L_out,
    const int chunk_start,
    const int chunk_size
) {
    const int tid = threadIdx.x;
    const int out_ch = blockIdx.x;
    const int local_pos = (blockIdx.y * blockDim.x + tid) * 4;
    const int out_pos = chunk_start + local_pos;
    const int batch_idx = blockIdx.z;

    if (out_pos >= chunk_start + chunk_size || out_pos >= L_out) return;

    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;

    float4 output = {0.0f, 0.0f, 0.0f, 0.0f};
    float* out_ptr = reinterpret_cast<float*>(&output);

    // Process input channels
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        const int in_ch = group_idx * group_size_in + local_in_ch;
        const float* weight_ptr = &w[out_ch * (group_size_in * K) + local_in_ch * K];

        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const float w_val = weight_ptr[k];
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (out_pos + i < L_out && out_pos + i < chunk_start + chunk_size) {
                    const int in_pos = (out_pos + i) * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        const float x_val = x[batch_idx * (C_in * L_in) + in_ch * L_in + in_pos];
                        out_ptr[i] += x_val * w_val;
                    }
                }
            }
        }
    }

    if (bias_ptr) {
        const float bias_val = bias_ptr[out_ch];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (out_pos + i < L_out && out_pos + i < chunk_start + chunk_size) {
                out_ptr[i] += bias_val;
            }
        }
    }

    // Store results
    const int out_offset = batch_idx * (C_out * L_out) + out_ch * L_out + out_pos;
    const int remaining = min(4, min(L_out - out_pos, chunk_start + chunk_size - out_pos));
    
    if (remaining == 4 && (reinterpret_cast<uintptr_t>(&y[out_offset]) & 15) == 0) {
        *reinterpret_cast<float4*>(&y[out_offset]) = output;
    } else {
        for (int i = 0; i < remaining; ++i) {
            y[out_offset + i] = out_ptr[i];
        }
    }
}

// Stream management class
class StreamManager {
public:
    StreamManager(int num_streams) {
        streams.resize(num_streams);
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }
    }

    ~StreamManager() {
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
    }

    cudaStream_t get_stream(int idx) { return streams[idx % streams.size()]; }
    int num_streams() const { return streams.size(); }

private:
    std::vector<cudaStream_t> streams;
};

at::Tensor conv1d_forward_impl_pipelined(
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
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Create stream manager with 4 streams
    StreamManager stream_mgr(4);

    // Calculate chunk size for each stream
    const int chunk_size = (L_out + stream_mgr.num_streams() - 1) / stream_mgr.num_streams();
    const int threads_per_block = 128;

    for (int chunk = 0; chunk < L_out; chunk += chunk_size) {
        const int current_chunk_size = std::min(chunk_size, static_cast<int>(L_out - chunk));
        const int stream_idx = (chunk / chunk_size) % stream_mgr.num_streams();
        cudaStream_t stream = stream_mgr.get_stream(stream_idx);

        dim3 block_dim(threads_per_block);
        dim3 grid_dim(
            C_out,
            (current_chunk_size + (threads_per_block * 4) - 1) / (threads_per_block * 4),
            N
        );

        conv1d_forward_kernel_pipelined<<<grid_dim, block_dim, 0, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, (int)C_in, (int)L_in, (int)C_out, (int)K,
            (int)stride, (int)padding, (int)dilation, (int)groups,
            (int)L_out, chunk, current_chunk_size
        );
    }

    // Synchronize all streams
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_pipelined failed: ", cudaGetErrorString(err));

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
            return conv1d_forward_impl_pipelined(x, weight, bias, stride, padding, dilation, groups);
        },
        "Pipelined 1D Convolution forward (CUDA)"
    );
}