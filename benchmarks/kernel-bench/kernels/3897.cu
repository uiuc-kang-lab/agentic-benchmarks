#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combined kernel: uses vectorized loads (float4) for aligned parts, and scalar processing for leftovers.
// This provides improved memory throughput while handling arbitrary tensor sizes.
__global__ void softsign_kernel_combined(const float* __restrict__ x, float* __restrict__ out, int num_elements, int num_vector4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process the main portion in groups of 4 (i.e. float4) if possible
    for (int i = tid; i < num_vector4; i += stride) {
        // Cast pointers to float4 for vectorized load/store
        const float4* in_vec = reinterpret_cast<const float4*>(x);
        float4* out_vec = reinterpret_cast<float4*>(out);
        float4 val = in_vec[i];
        
        // Compute Softsign activation component-wise
        val.x = val.x / (1.0f + fabsf(val.x));
        val.y = val.y / (1.0f + fabsf(val.y));
        val.z = val.z / (1.0f + fabsf(val.z));
        val.w = val.w / (1.0f + fabsf(val.w));

        out_vec[i] = val;
    }

    // Process any remaining elements that don't fit into a multiple of 4
    int remaining_start = num_vector4 * 4;
    for (int i = remaining_start + tid; i < num_elements; i += stride) {
        float v = x[i];
        out[i] = v / (1.0f + fabsf(v));
    }
}

// Host function invoked from Python / PyTorch
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Calculate how many elements can be processed in groups of 4
    int num_vector4 = num_elements / 4; // Each group processes 4 elements
    
    // Choose kernel launch configuration; using grid-stride loop so total elements is sufficient
    const int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    softsign_kernel_combined<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_elements, num_vector4);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined vectorized and fallback Softsign activation (CUDA)");
}
