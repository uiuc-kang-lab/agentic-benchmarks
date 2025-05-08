#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel leverages shared memory to preload chunks (tiles) of the input data from global memory.
// By loading a tile into shared memory, the kernel reduces global memory latency and performs the reduction
// within shared memory with lower latency and proper synchronization. Each block computes one output element.
__global__ void prod_reduce_shared_kernel(const float* __restrict__ input, float* __restrict__ output, int dim_size, int stride) {
    // Each block handles one output element along the non-reduced dimensions
    int out_idx = blockIdx.x;
    // Allocate shared memory tile for this block
    extern __shared__ float tile[];

    // Initialize the product as multiplicative identity
    float product = 1.0f;

    // Process the reduction dimension in tiles of blockDim.x elements
    for (int tile_start = 0; tile_start < dim_size; tile_start += blockDim.x) {
        int index = tile_start + threadIdx.x;
        // Load from global memory into shared memory if within bounds; else load 1.0f
        tile[threadIdx.x] = (index < dim_size) ? input[out_idx + index * stride] : 1.0f;
        __syncthreads();

        // Perform reduction within the tile using a binary tree reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                tile[threadIdx.x] *= tile[threadIdx.x + s];
            }
            __syncthreads();
        }
        // After the reduction, tile[0] holds the product of the current chunk
        if (threadIdx.x == 0) {
            product *= tile[0];
        }
        __syncthreads(); // Ensure all threads complete before next tile load
    }

    // Write the final product to global memory
    if (threadIdx.x == 0) {
        output[out_idx] = product;
    }
}

// Forward function: Removes the reduced dimension from the input shape and launches one block per output element
// It allocates shared memory for each block to store the tile used for the reduction.
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

    // Configure kernel launch: one block per output element; choose 256 threads per block
    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = threads * sizeof(float);

    prod_reduce_shared_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension using shared memory (CUDA)");
}
