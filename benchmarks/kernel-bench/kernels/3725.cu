#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Manual unrolled vectorized HardSigmoid kernel

template <typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    // Constants for HardSigmoid: y = clamp((x + 3)/6, 0, 1)
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);

    // Define vector type based on VEC_SIZE and scalar type
    // For float use float4 (VEC_SIZE = 4), for double use double2 (VEC_SIZE = 2)
    using vec_t = typename std::conditional<
                    std::is_same<scalar_t, float>::value,
                    float4,
                    typename std::conditional<std::is_same<scalar_t, double>::value,
                                              double2,
                                              void
                                             >::type
                  >::type;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes VEC_SIZE contiguous elements
    int base_idx = tid * VEC_SIZE;
    // Stride in terms of elements
    int stride = blockDim.x * gridDim.x * VEC_SIZE;

    for (int i = base_idx; i < numel; i += stride) {
        if (i + VEC_SIZE <= numel) {
            // Process full vector load/store
            vec_t v = *reinterpret_cast<const vec_t*>(&input[i]);
            if constexpr (VEC_SIZE == 4) {
                // Manually unroll for 4 elements (float case expected)
                scalar_t e0 = v.x;
                scalar_t e1 = v.y;
                scalar_t e2 = v.z;
                scalar_t e3 = v.w;

                // Compute (x+3)/6 using fused multiply-add
                if constexpr (std::is_same<scalar_t, float>::value) {
                    e0 = fmaf(e0, sixth, three * sixth);
                    e1 = fmaf(e1, sixth, three * sixth);
                    e2 = fmaf(e2, sixth, three * sixth);
                    e3 = fmaf(e3, sixth, three * sixth);

                    e0 = fmaxf(0.f, fminf(1.f, e0));
                    e1 = fmaxf(0.f, fminf(1.f, e1));
                    e2 = fmaxf(0.f, fminf(1.f, e2));
                    e3 = fmaxf(0.f, fminf(1.f, e3));
                } else {
                    e0 = fma(e0, sixth, three * sixth);
                    e1 = fma(e1, sixth, three * sixth);
                    e2 = fma(e2, sixth, three * sixth);
                    e3 = fma(e3, sixth, three * sixth);

                    e0 = fmax(0.0, fmin(1.0, e0));
                    e1 = fmax(0.0, fmin(1.0, e1));
                    e2 = fmax(0.0, fmin(1.0, e2));
                    e3 = fmax(0.0, fmin(1.0, e3));
                }
                v.x = e0; v.y = e1; v.z = e2; v.w = e3;
            } else if constexpr (VEC_SIZE == 2) {
                // Manually unroll for 2 elements (double case expected)
                scalar_t e0 = v.x;
                scalar_t e1 = v.y;

                if constexpr (std::is_same<scalar_t, float>::value) {
                    e0 = fmaf(e0, sixth, three * sixth);
                    e1 = fmaf(e1, sixth, three * sixth);

                    e0 = fmaxf(0.f, fminf(1.f, e0));
                    e1 = fmaxf(0.f, fminf(1.f, e1));
                } else {
                    e0 = fma(e0, sixth, three * sixth);
                    e1 = fma(e1, sixth, three * sixth);

                    e0 = fmax(0.0, fmin(1.0, e0));
                    e1 = fmax(0.0, fmin(1.0, e1));
                }
                v.x = e0; v.y = e1;
            }
            *reinterpret_cast<vec_t*>(&output[i]) = v;
        } else {
            // Residual scalar loop for remaining elements; unroll manually since VEC_SIZE is small
            int rem = numel - i;
            #pragma unroll
            for (int r = 0; r < VEC_SIZE; r++) {
                if (r < rem) {
                    scalar_t x = input[i + r];
                    if constexpr (std::is_same<scalar_t, float>::value) {
                        x = fmaf(x, sixth, three * sixth);
                        x = fmaxf(0.f, fminf(1.f, x));
                    } else {
                        x = fma(x, sixth, three * sixth);
                        x = fmax(0.0, fmin(1.0, x));
                    }
                    output[i + r] = x;
                }
            }
        }
    }
}


torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Select vector size: use 4 for float, 2 for double
    int vec_size = (input.scalar_type() == at::kFloat) ? 4 : 2;
    const int threads = 256;
    const int elements_per_block = threads * vec_size;
    const int blocks = (numel + elements_per_block - 1) / elements_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_kernel<scalar_t, /*VEC_SIZE=*/vec_size><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}
