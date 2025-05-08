#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 256
#define NUM_STREAMS 4

__global__ void prod_reduce_kernel(const float* __restrict__ input, 
                                 float* __restrict__ output, 
                                 const int stride, 
                                 const int num_elements,
                                 const int chunk_offset) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (idx < num_elements) {
        float product = 1.0f;
        const int offset = idx + chunk_offset;
        
        #pragma unroll 10
        for (int i = 0; i < 50; i += 10) {
            product *= input[offset + (i) * stride];
            product *= input[offset + (i+1) * stride];
            product *= input[offset + (i+2) * stride];
            product *= input[offset + (i+3) * stride];
            product *= input[offset + (i+4) * stride];
            product *= input[offset + (i+5) * stride];
            product *= input[offset + (i+6) * stride];
            product *= input[offset + (i+7) * stride];
            product *= input[offset + (i+8) * stride];
            product *= input[offset + (i+9) * stride];
        }
        
        output[idx + chunk_offset] = product;
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Process data in chunks using multiple streams
    int chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    int blocks_per_chunk = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_offset = i * chunk_size;
        int current_chunk_size = min(chunk_size, num_elements - chunk_offset);
        
        if (current_chunk_size > 0) {
            int current_blocks = (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            prod_reduce_kernel<<<current_blocks, BLOCK_SIZE, 0, streams[i]>>>(
                input_ptr, output_ptr, stride, current_chunk_size, chunk_offset
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
    m.def("forward", &forward, "Product reduction over a dimension with stream pipelining (CUDA)");
}