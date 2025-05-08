#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: templated exponential function
template <typename scalar_t>
__device__ __forceinline__ scalar_t my_exp(scalar_t x);

template <>
__device__ __forceinline__ float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ __forceinline__ double my_exp<double>(double x) {
    return exp(x);
}

// Generic element-wise SELU kernel for any floating point type
// (useful for double or fallback for other types)
template <typename scalar_t>
__global__ void selu_kernel_generic(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Iterate over elements with a grid-stride loop
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t activated = (x > static_cast<scalar_t>(0))
            ? x
            : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * activated;
    }
}

// Optimized vectorized kernel for float using float4 loads/stores
__global__ void selu_kernel_vectorized(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process the bulk of the data in groups of 4 using float4
    size_t num_vector_elems = numel / 4;  // number of float4 groups
    for (size_t i = idx; i < num_vector_elems; i += stride) {
        // reinterpret the data as float4
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        // Apply SELU: if x > 0 then x, else alpha * (exp(x)-1)
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : 1.67326324235437728481f * (expf(in_vec.x) - 1.0f);
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : 1.67326324235437728481f * (expf(in_vec.y) - 1.0f);
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : 1.67326324235437728481f * (expf(in_vec.z) - 1.0f);
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : 1.67326324235437728481f * (expf(in_vec.w) - 1.0f);
        
        // Multiply the result by the SELU scale
        out_vec.x *= 1.05070098735548049342f;
        out_vec.y *= 1.05070098735548049342f;
        out_vec.z *= 1.05070098735548049342f;
        out_vec.w *= 1.05070098735548049342f;
        
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle any remaining elements that weren't a multiple of 4
    size_t rem_start = num_vector_elems * 4;
    for (size_t i = rem_start + idx; i < numel; i += stride) {
        float x = input[i];
        float activated = (x > 0.0f) ? x : 1.67326324235437728481f * (expf(x) - 1.0f);
        output[i] = 1.05070098735548049342f * activated;
    }
}

// Unified forward function that dispatches to the optimal kernel based on tensor type
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    if (input.scalar_type() == torch::kFloat) {
        // For float tensors, leverage vectorized memory accesses using float4
        int threads = 256;
        int blocks = (numel/4 + threads - 1) / threads; 
        selu_kernel_vectorized<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), numel);
    } else {
        // Fallback generic kernel for other floating types (e.g., double)
        int threads = 1024;
        int blocks = (numel + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
            selu_kernel_generic<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel);
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "Combined SELU Activation Forward (CUDA)");
}
