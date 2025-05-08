#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

#define NUM_STREAMS 4
#define CHUNK_SIZE 1024

template <typename scalar_t>
__global__ void layernorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int chunk_offset) {

    int instance_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const scalar_t* in_ptr = input + (instance_idx + chunk_offset) * normalized_size;
    scalar_t* out_ptr = output + (instance_idx + chunk_offset) * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;

    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;
    
    #pragma unroll
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ accscalar_t mean;
    __shared__ accscalar_t inv_std;
    if (tid == 0) {
        mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
        accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
        inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    #pragma unroll
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        accscalar_t norm_val = (val - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) +
                                         static_cast<accscalar_t>(bias[i]));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threads = (normalized_size < 1024) ? normalized_size : 1024;
    int chunks_per_stream = (outer_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        int shared_size = threads * 2 * sizeof(accscalar_t);
        
        for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
            int chunk_offset = stream_idx * chunks_per_stream;
            int chunk_size = std::min(chunks_per_stream, outer_size - chunk_offset);
            
            if (chunk_size <= 0) continue;
            
            layernorm_forward_kernel<scalar_t><<<chunk_size, threads, shared_size, streams[stream_idx]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                static_cast<float>(eps),
                output.data_ptr<scalar_t>(),
                normalized_size,
                chunk_offset);
        }
    }));

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}