#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

static const int NUM_STREAMS = 4;
static cudaStream_t streams[NUM_STREAMS];
static bool streams_created = false;

void create_streams() {
    if (!streams_created) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        streams_created = true;
    }
}

void destroy_streams() {
    if (streams_created) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        streams_created = false;
    }
}

template <typename scalar_t>
__global__ void layernorm_streamed_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int chunk_size,
    const int chunk_offset) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int instance_idx = blockIdx.x + chunk_offset;
    
    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x * blockDim.y;
    
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;
    
    const int thread_stride = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;
    
    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;
    
    #pragma unroll 8
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[thread_id] = local_sum;
    s_sum_sq[thread_id] = local_sum_sq;
    __syncthreads();
    
    if (thread_id < 32) {
        accscalar_t warp_sum = 0;
        accscalar_t warp_sum_sq = 0;
        
        #pragma unroll
        for (int i = thread_id; i < thread_stride; i += 32) {
            warp_sum += s_sum[i];
            warp_sum_sq += s_sum_sq[i];
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
        }
        
        if (thread_id == 0) {
            s_sum[0] = warp_sum;
            s_sum_sq[0] = warp_sum_sq;
        }
    }
    __syncthreads();
    
    __shared__ accscalar_t mean, inv_std;
    if (thread_id == 0) {
        mean = s_sum[0] / normalized_size;
        accscalar_t variance = (s_sum_sq[0] / normalized_size) - (mean * mean);
        inv_std = rsqrt(variance + static_cast<accscalar_t>(eps));
    }
    __syncthreads();
    
    #pragma unroll 8
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t normalized = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(
            normalized * static_cast<accscalar_t>(weight[idx]) + 
            static_cast<accscalar_t>(bias[idx]));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    create_streams();
    
    auto output = torch::empty_like(x);
    
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;
    const int chunk_size = (outer_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    const dim3 threads(32, 32);
    const int shared_mem_size = threads.x * threads.y * 2 * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int stream_chunk_size = std::min(chunk_size, outer_size - i * chunk_size);
            if (stream_chunk_size <= 0) break;
            
            const dim3 blocks(stream_chunk_size);
            
            layernorm_streamed_kernel<scalar_t><<<blocks, threads, shared_mem_size, streams[i]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                static_cast<float>(eps),
                output.data_ptr<scalar_t>(),
                normalized_size,
                chunk_size,
                i * chunk_size);
        }
    }));
    
    // Synchronize all streams before returning
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
    // Add cleanup function for streams
    m.def("cleanup", &destroy_streams, "Cleanup CUDA streams");
}