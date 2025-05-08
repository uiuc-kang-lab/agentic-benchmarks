#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <vector>

namespace py = pybind11;

#define TILE_SIZE 32
#define BLOCK_SIZE 256

__global__ void optimized_conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    int start_n,
    int N_chunk,
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
    __shared__ float s_input[TILE_SIZE][BLOCK_SIZE + K - 1];
    __shared__ float s_weight[TILE_SIZE][K];
    
    int out_ch = blockIdx.x;
    int out_pos = blockIdx.y * blockDim.x + threadIdx.x;
    int n_local = blockIdx.z;
    
    if (out_pos >= L_out || n_local >= N_chunk) return;
    
    int n = start_n + n_local;
    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;
    
    float result = 0.0f;
    
    for (int tile = 0; tile < group_size_in; tile += TILE_SIZE) {
        int valid_channels = min(TILE_SIZE, group_size_in - tile);
        
        for (int i = threadIdx.x; i < valid_channels * (BLOCK_SIZE + K - 1); i += blockDim.x) {
            int local_in_ch = i / (BLOCK_SIZE + K - 1);
            int local_pos = i % (BLOCK_SIZE + K - 1);
            
            if (local_in_ch < valid_channels) {
                int in_ch = group_idx * group_size_in + tile + local_in_ch;
                int in_pos = (blockIdx.y * blockDim.x + local_pos) * stride - padding;
                
                float val = 0.0f;
                if (in_pos >= 0 && in_pos < L_in) {
                    val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                }
                s_input[local_in_ch][local_pos] = val;
            }
        }
        
        for (int i = threadIdx.x; i < valid_channels * K; i += blockDim.x) {
            int local_in_ch = i / K;
            int k = i % K;
            
            if (local_in_ch < valid_channels) {
                s_weight[local_in_ch][k] = w[out_ch * (group_size_in * K) + 
                                           (tile + local_in_ch) * K + k];
            }
        }
        
        __syncthreads();
        
        for (int local_in_ch = 0; local_in_ch < valid_channels; local_in_ch++) {
            for (int k = 0; k < K; k++) {
                int input_idx = threadIdx.x + k * dilation;
                result += s_input[local_in_ch][input_idx] * 
                         s_weight[local_in_ch][k];
            }
        }
        
        __syncthreads();
    }
    
    if (bias_ptr) {
        result += bias_ptr[out_ch];
    }
    
    if (out_pos < L_out) {
        y[n * (C_out * L_out) + out_ch * L_out + out_pos] = result;
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
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "Inputs must be float32");

    auto x_sizes = x.sizes();
    int64_t N = x_sizes[0], C_in = x_sizes[1], L_in = x_sizes[2];
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0], K = w_sizes[2];
    
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length");

    auto y = torch::empty({N, C_out, L_out}, x.options());
    
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    int num_streams = std::min(4, (int)N);
    int chunk_size = (N + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(C_out, (L_out + block.x - 1) / block.x, chunk_size);

    for (int i = 0; i < num_streams; i++) {
        int start_n = i * chunk_size;
        if (start_n >= N) break;
        
        int current_chunk = std::min(chunk_size, (int)(N - start_n));
        grid.z = current_chunk;

        optimized_conv1d_kernel<<<grid, block, 0, streams[i]>>>(x.data_ptr<float>(),
            weight.data_ptr<float>(), bias_ptr, y.data_ptr<float>(),
            start_n, current_chunk, C_in, L_in, C_out, K,
            stride, padding, dilation, groups, L_out);
    }

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", 
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
              return conv1d_forward_impl(x, weight, bias, 
                                       stride, padding, dilation, groups);
          },
          "Optimized 1D Convolution with shared memory and stream pipelining");
}