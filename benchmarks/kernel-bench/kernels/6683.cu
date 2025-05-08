#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that assumes the reduction dimension is the innermost (contiguous) dimension.
// Each block computes the product reduction for one output element. Global memory accesses are coalesced
// because the reduction values are stored contiguously. Partial products are computed in registers
// and then reduced using shared memory.

__global__ void coalesced_product_reduction_kernel(const float* __restrict__ input,
                                                     float* __restrict__ output,
                                                     int N) {
    // Each block handles one output element corresponding to a contiguous segment of length N
    int out_idx = blockIdx.x;
    const float* base = input + out_idx * N;

    int tid = threadIdx.x;
    float prod = 1.0f;
    // Loop over the reduction dimension in a way that consecutive threads in a warp read consecutive elements
    for (int i = tid; i < N; i += blockDim.x) {
        prod *= base[i];
    }

    extern __shared__ float sdata[];
    sdata[tid] = prod;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = sdata[0];
    }
}

// Forward function: performs product reduction over the specified dimension while ensuring coalesced
// memory accesses. If the reduction dimension is not the innermost dimension, the input tensor is
// permuted so that the reduction dimension becomes contiguous (i.e. the last dimension), ensuring that
// global loads by threads in a warp are from consecutive memory locations.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    int orig_dim = dim;
    // If the reduction dimension is not the innermost, permute the tensor to bring it to the last dimension.
    if (dim != (x.dim() - 1)) {
        std::vector<int64_t> order;
        for (int i = 0; i < x.dim(); ++i) {
            if (i != dim)
                order.push_back(i);
        }
        order.push_back(dim);
        x = x.permute(order).contiguous();
        // Now the reduction dimension is the last one
        dim = x.dim() - 1;
    }

    // The size of the reduction (innermost) dimension
    int N = x.size(dim);
    
    // The output shape is the input shape with the last dimension (reduction dimension) removed.
    auto sizes = x.sizes().vec();
    sizes.pop_back();
    torch::Tensor output = torch::empty(sizes, x.options());

    // Number of output elements equals the product of dimensions except the reduction dim
    int num_outputs = output.numel();
    
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int threads = 256;  // Number of threads per block
    int blocks = num_outputs; // One block per output element
    int shared_mem = threads * sizeof(float);

    coalesced_product_reduction_kernel<<<blocks, threads, shared_mem>>>(input_ptr, output_ptr, N);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced product reduction over a dimension (CUDA)");
}
