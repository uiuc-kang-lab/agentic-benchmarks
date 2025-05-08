#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optional: toggle warp-level reduction with this macro. Set to 0 to disable.
#ifndef USE_WARP_REDUCTION
#define USE_WARP_REDUCTION 1
#endif

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

// Fused device function that computes the SELU activation, optionally using warp-level reduction.
// The warp-level reduction here can be used for auxiliary computations (e.g. debugging or statistics)
// without affecting the SELU output. When disabled, it defaults to a straightforward computation.

template <typename scalar_t>
__device__ inline scalar_t selu_activate(scalar_t x) {
    // SELU parameters
    const scalar_t alpha  = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Compute SELU activation: lambda * (x if x > 0 else alpha * (exp(x) - 1))
    scalar_t pos_val = x;
    scalar_t neg_val = alpha * (my_exp(x) - static_cast<scalar_t>(1));
    scalar_t cond = static_cast<scalar_t>(x > static_cast<scalar_t>(0));
    scalar_t value = cond * pos_val + (static_cast<scalar_t>(1) - cond) * neg_val;
    scalar_t selu_val = lambda * value;

#if USE_WARP_REDUCTION
    // Optionally perform a warp-level reduction as a dummy auxiliary computation.
    unsigned lane = threadIdx.x % warpSize;
    unsigned mask = 0xffffffff;
    scalar_t warp_sum = selu_val;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(mask, warp_sum, offset);
    }
    // Dummy usage: if the lane is 0, perform an identity operation with warp_sum
    if (lane == 0 && warp_sum != static_cast<scalar_t>(0)) {
        selu_val *= (warp_sum / warp_sum);
    }
#endif

    return selu_val;
}

// Fused SELU CUDA kernel: combines SELU computation with optional warp-level reduction and
// leverages __ldg for read-only global memory loads to improve cache efficiency.

template <typename scalar_t>
__global__ void selu_kernel_fused(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    size_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Use __ldg for better read-only memory caching
        scalar_t x = __ldg(&input[idx]);
        output[idx] = selu_activate(x);
    }
}

// Host function launching the fused SELU kernel. Exposed to Python as "forward".

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_fused<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "Fused SELU Activation Forward (CUDA)");
}
