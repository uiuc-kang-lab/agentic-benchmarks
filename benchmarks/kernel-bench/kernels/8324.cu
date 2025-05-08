#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <vector>

namespace py = pybind11;

__global__ void conv1d_forward_kernel_streamed(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    const int N_chunk,    // chunk size for batch dimension
    const int C_in,
    const int L_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out,
    const int chunk_offset // offset in the batch dimension
) {
    extern __shared__ float shw[];
    
    int thread_out = threadIdx.x;
    int out_ch = blockIdx.y;
    int n = blockIdx.x;
    
    // Adjust n to account for chunk offset
    n += chunk_offset;
    
    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;
    
    // Load weights into shared memory
    for (int i = threadIdx.x; i < group_size_in * K; i += blockDim.x) {
        shw[i] = w[out_ch * (group_size_in * K) + i];
    }
    __syncthreads();
    
    for (int out_pos = thread_out; out_pos < L_out; out_pos += blockDim.x) {
        float sum = 0.0f;
        for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
            int in_ch = group_idx * group_size_in + local_in_ch;
            for (int k = 0; k < K; k++) {
                int in_pos = out_pos * stride + k * dilation - padding;
                int clamped_in_pos = in_pos < 0 ? 0 : (in_pos >= L_in ? (L_in - 1) : in_pos);
                float mask = ((unsigned)in_pos < (unsigned)L_in) ? 1.0f : 0.0f;
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + clamped_in_pos];
                sum += mask * x_val * shw[local_in_ch * K + k];
            }
        }
        
        if (bias_ptr) {
            sum += bias_ptr[out_ch];
        }
        
        y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
    }
}

at::Tensor conv1d_forward_impl_streamed(
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
    
    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));
    
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }
    
    // Create CUDA streams
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate chunk size for batch dimension
    int chunk_size = (N + num_streams - 1) / num_streams;
    int sharedMemSize = (C_in / groups) * K * sizeof(float);
    
    // Process data in chunks using multiple streams
    for (int chunk = 0; chunk < N; chunk += chunk_size) {
        int current_chunk_size = std::min(chunk_size, static_cast<int>(N - chunk));
        int stream_idx = (chunk / chunk_size) % num_streams;
        
        dim3 grid(current_chunk_size, C_out);
        int threads = std::min(256, static_cast<int>(L_out));
        
        conv1d_forward_kernel_streamed<<<grid, threads, sharedMemSize, streams[stream_idx]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            current_chunk_size,
            C_in, L_in, C_out, K,
            stride, padding, dilation, groups, L_out,
            chunk
        );
    }
    
    // Synchronize all streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
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
            return conv1d_forward_impl_streamed(x, weight, bias, stride, padding, dilation, groups);
        },
        "Streamed 1D Convolution forward (CUDA) with computation-memory overlap"
    );
}