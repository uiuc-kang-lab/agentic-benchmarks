#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// This kernel distributes workloads evenly by calculating the number of elements each thread should process
// and ensuring that all threads have a similar amount of work.

template <typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);

    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double2
    >::type;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    size_t elements_per_thread = (numel + total_threads - 1) / total_threads;
    size_t start_idx = tid * elements_per_thread;
    size_t end_idx = min(start_idx + elements_per_thread, numel);

    for (size_t i = start_idx; i < end_idx; i += VEC_SIZE) {
        vec_t chunk = *reinterpret_cast<const vec_t*>(&input[i]);
        scalar_t elems[VEC_SIZE];
        *reinterpret_cast<vec_t*>(elems) = chunk;

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            scalar_t x = elems[j];
            x = (x + three) * sixth;  // computes (x + 3) / 6
            x = (x < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) :
                (x > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : x);
            elems[j] = x;
        }

        *reinterpret_cast<vec_t*>(&output[i]) = *reinterpret_cast<vec_t*>(elems);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "even_workload_hardsigmoid_cuda", ([&] {
        constexpr int vec_size = std::is_same<scalar_t, float>::value ? 4 : 2;
        int blocks = (numel + threads * vec_size - 1) / (threads * vec_size);
        hardsigmoid_kernel<scalar_t, vec_size><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Even Workload Distributed HardSigmoid activation forward (CUDA)");
}
