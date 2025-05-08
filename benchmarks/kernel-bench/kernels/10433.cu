#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define NUM_STREAMS 4

__global__ void cumsum_kernel(const float* input, float* output, int dim, int outer_size, int inner_size, int stride, int chunk_offset) {
    int outer_idx = blockIdx.x + chunk_offset;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        int base_idx = outer_idx * stride * inner_size + inner_idx;
        
        #pragma unroll 16
        for (int i = 0; i < stride; ++i) {
            int idx = base_idx + i * inner_size;
            sum += input[idx];
            output[idx] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate chunk size for each stream
    int chunk_size = (outer_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_offset = i * chunk_size;
        int current_chunk_size = min(chunk_size, outer_size - chunk_offset);
        
        if (current_chunk_size > 0) {
            cumsum_kernel<<<current_chunk_size, inner_size, 0, streams[i]>>>(
                x.data_ptr<float>(), 
                output.data_ptr<float>(), 
                dim, 
                outer_size, 
                inner_size, 
                stride,
                chunk_offset
            );
        }
    }

    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with stream overlap");
}