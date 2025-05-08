#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

// This kernel computes the product reduction for each output element using one block per output.
// It minimizes synchronization by calling __syncthreads() only during the shared memory reduction until
// the number of active threads reaches a warp (32), after which warp-level primitives are used.
__global__ void prod_reduction_kernel(const float* input, float* output, int dim_size, int stride, int num_elements) {
    // Each block corresponds to one output element
    int outIdx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // Each thread computes a partial product for its assigned indices in the reduction dimension.
    float partial = 1.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        partial *= input[outIdx + i * stride];
    }
    sdata[tid] = partial;
    __syncthreads(); // Ensure all partial products are written to shared memory

    int blockSize = blockDim.x;
    // Reduce in shared memory, synchronizing only when necessary
    for (int s = blockSize / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // Final reduction within a warp using warp-level shuffle intrinsics (no __syncthreads needed)
    if (tid < WARP_SIZE) {
        float warpVal = sdata[tid];
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            warpVal *= __shfl_down_sync(0xffffffff, warpVal, offset);
        }
        if (tid == 0) {
            output[outIdx] = warpVal;
        }
    }
}

// Forward function: removes the reduction dimension from the output shape and launches one block per output element.
// This function uses dynamic shared memory for intra-block reduction and minimizes synchronizations to those required.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Prepare output shape by removing the reduced dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch one block per output element, with an appropriate number of threads per block
    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = threads * sizeof(float);

    prod_reduction_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}
