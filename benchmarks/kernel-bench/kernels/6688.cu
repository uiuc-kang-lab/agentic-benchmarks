#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combined optimized kernel performing product reduction over a specified dimension.
// It merges manual loop unrolling for the accumulation and shared memory reduction with
// warp-level reduction using shuffle intrinsics. This minimizes synchronization overhead
// and loop iterations, leveraging the best ideas from both original kernels.
__global__ void combined_prod_reduce_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int dim_size,
                                             int stride) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int out_idx = blockIdx.x;  // Each block handles one output element

    // Compute partial product with loop unrolling hint
    float prod = 1.0f;
    #pragma unroll
    for (int i = tid; i < dim_size; i += blockDim.x) {
        prod *= input[out_idx + i * stride];
    }
    sdata[tid] = prod;
    __syncthreads();

    // Shared memory reduction with manual unrolling to reduce loop overhead
    if (blockDim.x >= 512) {
        if (tid < 256) {
            sdata[tid] *= sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sdata[tid] *= sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            sdata[tid] *= sdata[tid + 64];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle intrinsics; no synchronization needed within a warp
    if (tid < 32) {
        float warp_val = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_val *= __shfl_down_sync(0xffffffff, warp_val, offset);
        }
        if (tid == 0) {
            output[out_idx] = warp_val;
        }
    }
}

// Forward function: sets up the output tensor by removing the reduced dimension and
// launches one kernel block per output element.
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

    // Kernel configuration
    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = threads * sizeof(float);

    combined_prod_reduce_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction over a dimension (CUDA)");
}
