#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

template <typename scalar_t>
__global__ void l2_norm_reduce_kernel_streamed(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int vectors_per_stream,
    const int stride_C,
    const int outer_stride) {
    
    const int vec_idx = blockIdx.x + (blockIdx.y * vectors_per_stream);
    if (vec_idx >= vectors_per_stream) return;
    
    const int base_offset = vec_idx * outer_stride;
    scalar_t sum = 0;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }
    
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        norms[vec_idx] = sum;
    }
}

template <typename scalar_t>
__global__ void l2_norm_normalize_kernel_streamed(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int vectors_per_stream,
    const int stride_C,
    const int outer_stride) {
    
    const int vec_idx = blockIdx.x + (blockIdx.y * vectors_per_stream);
    if (vec_idx >= vectors_per_stream) return;
    
    const int base_offset = vec_idx * outer_stride;
    const scalar_t inv_norm = rsqrt(norms[vec_idx] + 1e-12);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const int idx = base_offset + i * stride_C;
        output[idx] = input[idx] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int num_streams = 4;  // Number of CUDA streams to use
    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int vectors_per_stream = (total_vectors + num_streams - 1) / num_streams;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto options = torch::TensorOptions().device(input.device()).pinned_memory(true);
    auto norms = torch::zeros({total_vectors}, options);

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 512;
    const int blocks_per_stream = (vectors_per_stream + 31) / 32;
    dim3 grid(32, blocks_per_stream);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_streamed", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int stream_offset = i * vectors_per_stream;
            auto input_view = input.narrow(0, stream_offset, 
                std::min(vectors_per_stream, total_vectors - stream_offset));
            auto output_view = output.narrow(0, stream_offset,
                std::min(vectors_per_stream, total_vectors - stream_offset));
            auto norms_view = norms.narrow(0, stream_offset,
                std::min(vectors_per_stream, total_vectors - stream_offset));

            l2_norm_reduce_kernel_streamed<scalar_t><<<grid, threads, 0, streams[i]>>>(
                input_view.data_ptr<scalar_t>(),
                norms_view.data_ptr<scalar_t>(),
                C,
                vectors_per_stream,
                stride_C,
                outer_stride
            );

            l2_norm_normalize_kernel_streamed<scalar_t><<<grid, threads, 0, streams[i]>>>(
                input_view.data_ptr<scalar_t>(),
                output_view.data_ptr<scalar_t>(),
                norms_view.data_ptr<scalar_t>(),
                C,
                vectors_per_stream,
                stride_C,
                outer_stride
            );
        }
    }));

    // Synchronize and cleanup streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with stream overlap");
}