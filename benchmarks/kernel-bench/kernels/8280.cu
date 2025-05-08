#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

__global__ void conv1d_forward_kernel_hybrid(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    const int N, const int C_in, const int L_in,
    const int C_out, const int K, const int stride,
    const int padding, const int dilation, const int groups,
    const int L_out, const int segment_size
) {
    int out_ch = blockIdx.x;
    int segment = blockIdx.y;
    int local_pos = threadIdx.x;
    int n = blockIdx.z;
    
    int out_pos = segment * segment_size + local_pos;
    if (out_pos >= L_out) return;

    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;

    extern __shared__ float shmem[];
    float* shared_weights = shmem;
    float* shared_input = &shmem[group_size_in * K];

    int total_weights = group_size_in * K;
    for (int i = threadIdx.x; i < total_weights; i += blockDim.x) {
        shared_weights[i] = w[out_ch * total_weights + i];
    }

    int input_offset = n * (C_in * L_in);
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                shared_input[local_in_ch * K + k] = 
                    x[input_offset + in_ch * L_in + in_pos];
            } else {
                shared_input[local_in_ch * K + k] = 0.0f;
            }
        }
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            sum += shared_input[local_in_ch * K + k] * 
                   shared_weights[local_in_ch * K + k];
        }
    }

    if (bias_ptr) {
        sum += bias_ptr[out_ch];
    }

    if (out_pos < L_out) {
        y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
    }
}

at::Tensor conv1d_forward_impl_hybrid(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto x_sizes = x.sizes();
    int64_t N = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];
    
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K = w_sizes[2];
    
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    
    auto y = torch::empty({N, C_out, L_out}, x.options());
    
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    const int segment_size = 128;
    int num_segments = (L_out + segment_size - 1) / segment_size;
    
    dim3 block(256);
    dim3 grid(C_out, num_segments, N);
    
    int group_size_in = C_in / groups;
    size_t shared_mem_size = (group_size_in * K * sizeof(float)) * 2;
    
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < N; i++) {
        cudaStream_t& stream = streams[i % num_streams];
        dim3 grid_batch(C_out, num_segments, 1);
        
        conv1d_forward_kernel_hybrid<<<grid_batch, block, shared_mem_size, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, (int)C_in, (int)L_in, (int)C_out, (int)K,
            (int)stride, (int)padding, (int)dilation, (int)groups,
            (int)L_out, segment_size
        );
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv1d_forward_impl_hybrid,
        "Optimized 1D Convolution forward (CUDA) using hybrid approach");
}