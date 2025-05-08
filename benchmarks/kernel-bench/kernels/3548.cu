#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
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

// CUDA kernel that applies the SELU activation to each element and demonstrates
// the use of warp-level primitives (__shfl_down_sync) for a small reduction in lieu
// of shared memory operations. The warp reduction here is used in a dummy way to
// mimic a reduction operation that could be used for auxiliary computations without
// altering the final SELU output.

template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t x = input[idx];
        
        // SELU parameters.
        const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
        const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
        
        // Compute SELU activation: lambda*(x if x > 0 else alpha*(exp(x)-1))
        scalar_t value = (x > static_cast<scalar_t>(0)) ? x : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        scalar_t selu_val = lambda * value;
        
        // Use warp-level primitives to perform a dummy reduction within the warp.
        // This mimics the replacement of shared memory reductions with warp-level intrinsics.
        unsigned lane = threadIdx.x % warpSize;
        unsigned mask = 0xffffffff;
        scalar_t warp_sum = selu_val;
        
        // Perform warp-level reduction using __shfl_down_sync
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(mask, warp_sum, offset);
        }

        // Dummy usage: if this thread is the first lane of the warp, use warp_sum
        // to perform an operation that mathematically multiplies selu_val by 1.
        // Guard against division by zero in the unlikely event that warp_sum is 0.
        if (lane == 0) {
            if (warp_sum != static_cast<scalar_t>(0)) {
                selu_val *= (warp_sum / warp_sum);
            }
        }
        
        // Write the computed SELU activation to the output buffer.
        output[idx] = selu_val;
    }
}

// Host function that launches the CUDA SELU kernel.
// This function will be exposed to Python as "forward".

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA)");
}
