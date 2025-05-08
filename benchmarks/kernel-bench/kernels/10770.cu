#include <torch/extension.h>
#include <vector>

template<typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* input, scalar_t* output, int64_t dim_size, int64_t stride_dim, int64_t num_slices) {
    extern __shared__ char smem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(smem);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < num_slices) {
        // Load the slice into shared memory (each thread handles its own slice)
        for (int i = 0; i < dim_size; i++) {
            tile[threadIdx.x * dim_size + i] = input[idx * dim_size + i * stride_dim];
        }
        __syncthreads();
        
        // Compute reverse cumulative sum in shared memory
        scalar_t sum = 0;
        for (int i = dim_size - 1; i >= 0; i--) {
            sum += tile[threadIdx.x * dim_size + i];
            tile[threadIdx.x * dim_size + i] = sum;
        }
        __syncthreads();
        
        // Write the computed tile back to global memory
        for (int i = 0; i < dim_size; i++) {
            output[idx * dim_size + i * stride_dim] = tile[threadIdx.x * dim_size + i];
        }
        idx += gridDim.x * blockDim.x;
    }
}

at::Tensor reverse_cumsum_optimized(at::Tensor x, int64_t dim) {
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    auto output = torch::empty_like(x);
    const int64_t dim_size = x.size(dim);
    const int64_t num_slices = x.numel() / dim_size;
    const int64_t stride_dim = x.stride(dim);

    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int64_t slices_per_stream = (num_slices + num_streams - 1) / num_streams;
    
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum", [&] {
        for (int i = 0; i < num_streams; ++i) {
            const int64_t start = i * slices_per_stream;
            const int64_t end = std::min((i + 1) * slices_per_stream, num_slices);
            const int64_t stream_slices = end - start;
            if (stream_slices <= 0) continue;

            const scalar_t* slice_in = x.data_ptr<scalar_t>() + start * dim_size;
            scalar_t* slice_out = output.data_ptr<scalar_t>() + start * dim_size;
            
            const int threads = 256;
            const int blocks = (stream_slices + threads - 1) / threads;

            reverse_cumsum_kernel<scalar_t>
                <<<blocks, threads, 0, streams[i]>>>(
                    slice_in, slice_out, dim_size, stride_dim, stream_slices
                );
        }
    });

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_optimized, "Multi-stream reverse cumsum (CUDA)");
}
