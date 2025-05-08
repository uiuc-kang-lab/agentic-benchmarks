#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential for float and double
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}


// Vectorized SELU kernel using 128-bit aligned loads/stores and __ldg() for read-only accesses
// For float, we use float4 (4 x 32-bit = 128-bit) and for double, double2 (2 x 64-bit = 128-bit).

template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Determine vector size: 4 elements for float and 2 for double to achieve 128-bit access
    const int vecSize = (sizeof(scalar_t) == 4 ? 4 : 2);
    size_t numVec = numel / vecSize;

    // Process vectorized portion
    if (sizeof(scalar_t) == 4) {
        using Vec = float4;
        Vec* outVecPtr = reinterpret_cast<Vec*>(output);
        const Vec* inVecPtr = reinterpret_cast<const Vec*>(input);
        for (size_t i = tid; i < numVec; i += stride) {
            // Use __ldg() to load from global memory (read-only cache)
            Vec in_vec = __ldg(&inVecPtr[i]);
            Vec out_vec;
            // Apply SELU element-wise
            out_vec.x = 1.05070098735548049342f * ((in_vec.x > 0.f) ? in_vec.x : 1.67326324235437728481f * (expf(in_vec.x) - 1.f));
            out_vec.y = 1.05070098735548049342f * ((in_vec.y > 0.f) ? in_vec.y : 1.67326324235437728481f * (expf(in_vec.y) - 1.f));
            out_vec.z = 1.05070098735548049342f * ((in_vec.z > 0.f) ? in_vec.z : 1.67326324235437728481f * (expf(in_vec.z) - 1.f));
            out_vec.w = 1.05070098735548049342f * ((in_vec.w > 0.f) ? in_vec.w : 1.67326324235437728481f * (expf(in_vec.w) - 1.f));
            outVecPtr[i] = out_vec;
        }
    } else {
        using Vec = double2;
        Vec* outVecPtr = reinterpret_cast<Vec*>(output);
        const Vec* inVecPtr = reinterpret_cast<const Vec*>(input);
        for (size_t i = tid; i < numVec; i += stride) {
            Vec in_vec = __ldg(&inVecPtr[i]);
            Vec out_vec;
            out_vec.x = 1.05070098735548049342 * ((in_vec.x > 0.0) ? in_vec.x : 1.67326324235437728481 * (exp(in_vec.x) - 1.0));
            out_vec.y = 1.05070098735548049342 * ((in_vec.y > 0.0) ? in_vec.y : 1.67326324235437728481 * (exp(in_vec.y) - 1.0));
            outVecPtr[i] = out_vec;
        }
    }

    // Process any remaining elements that don't fit into a full vector load/store
    size_t remStart = numVec * vecSize;
    for (size_t i = remStart + tid; i < numel; i += stride) {
        scalar_t x = __ldg(&input[i]);
        scalar_t res = (x > static_cast<scalar_t>(0))
                           ? x
                           : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function exposed to Python via pybind11
// Launches the vectorized SELU kernel

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "Vectorized SELU Activation Forward (CUDA)");
}
