#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Define vectorized traits for float and double to enable vectorized load/stores

template <typename scalar_t>
struct vectorized_traits;

// Specialization for float using float4 (4 floats, 16 bytes) -> enhances memory coalescing
template <>
struct vectorized_traits<float> {
    using vec_t = float4;
    static const int width = 4;
    __device__ static void apply(const vec_t &in, vec_t &out) {
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
    }
};

// Specialization for double using double2 (2 doubles, 16 bytes)
template <>
struct vectorized_traits<double> {
    using vec_t = double2;
    static const int width = 2;
    __device__ static void apply(const vec_t &in, vec_t &out) {
        out.x = tanh(in.x);
        out.y = tanh(in.y);
    }
};

// Kernel that uses vectorized loads/stores to improve memory coalescing
template <typename scalar_t>
__global__ void tanh_kernel_coalesced(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int size) {

    using traits = vectorized_traits<scalar_t>;
    using vec_t = typename traits::vec_t;
    constexpr int vec_width = traits::width;

    // Determine how many full vectorized loads we can do
    int num_vec = size / vec_width;
    int remainder = size % vec_width;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process vectorized portion: each load/store handles vec_width elements
    for (int i = idx; i < num_vec; i += stride) {
        vec_t vec_in = reinterpret_cast<const vec_t*>(input)[i];
        vec_t vec_out;
        traits::apply(vec_in, vec_out);
        reinterpret_cast<vec_t*>(output)[i] = vec_out;
    }

    // Process any remaining elements (if size is not a multiple of vec_width)
    // This section is executed by one thread since the remainder is minimal
    if (idx == 0) {
        int offset = num_vec * vec_width;
        for (int i = 0; i < remainder; i++) {
            // Use if constexpr to select the correct precision function
            if constexpr (std::is_same<scalar_t, float>::value) {
                output[offset + i] = tanhf(input[offset + i]);
            } else {
                output[offset + i] = tanh(input[offset + i]);
            }
        }
    }
}

// Host function to launch the CUDA kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_coalesced", ([&] {
        // Determine the number of vectorized iterations based on the type
        constexpr int vec_width = vectorized_traits<scalar_t>::width;
        int num_vec = size / vec_width;
        int blocks = (num_vec + threads - 1) / threads;
        if (blocks == 0) blocks = 1;  // Ensure at least one block
        tanh_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized and coalesced Tanh forward (CUDA)");
}
