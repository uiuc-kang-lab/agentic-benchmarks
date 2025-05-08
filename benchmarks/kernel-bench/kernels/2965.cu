#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Specialize vectorized traits for float and double to use vectorized loads and stores

template <typename scalar_t>
struct vectorized_traits;

// For float: use float4 (16 bytes, 4 floats)
template <>
struct vectorized_traits<float> {
    using vec_t = float4;
    static const int width = 4;
    __device__ static void apply(const vec_t &in, vec_t &out) {
        // Use device-specific fast tanh function for floats
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
    }
};

// For double: use double2 (16 bytes, 2 doubles)
template <>
struct vectorized_traits<double> {
    using vec_t = double2;
    static const int width = 2;
    __device__ static void apply(const vec_t &in, vec_t &out) {
        // Use device-specific tanh function for doubles
        out.x = tanh(in.x);
        out.y = tanh(in.y);
    }
};

// The kernel combines vectorized processing for the bulk of the data and a grid-stride loop
// for any remaining elements. This leverages efficient memory coalescing while keeping
// the code simple and reducing warp divergence in the tail processing.

template <typename scalar_t>
__global__ void tanh_vectorized_combined_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int size) {

    using traits = vectorized_traits<scalar_t>;
    using vec_t = typename traits::vec_t;
    constexpr int vec_width = traits::width;

    // Determine how many complete vectorized groups we can process
    int num_vec = size / vec_width;
    int rem_start = num_vec * vec_width;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process vectorized portion using grid-stride loop
    for (int i = idx; i < num_vec; i += stride) {
        // Reinterpret the input pointer as pointer to vec_t
        vec_t in_vec = reinterpret_cast<const vec_t*>(input)[i];
        vec_t out_vec;
        traits::apply(in_vec, out_vec);
        reinterpret_cast<vec_t*>(output)[i] = out_vec;
    }

    // Process remaining elements with a scalar loop
    for (int i = rem_start + idx; i < size; i += stride) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            output[i] = tanhf(input[i]);
        } else {
            output[i] = tanh(input[i]);
        }
    }
}

// Host function that dispatches the kernel

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int max_blocks = 65535;
    
    // Use a simple heuristic: calculate the number of blocks based on total elements
    int blocks = (size + threads - 1) / threads;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_vectorized_combined", ([&] {
        using traits = vectorized_traits<scalar_t>;
        // Optionally, optimize block count using the vectorized portion size
        int num_vec = size / traits::width;
        int opt_blocks = (num_vec + threads - 1) / threads;
        if (opt_blocks < blocks) {
            blocks = opt_blocks;
        }
        
        tanh_vectorized_combined_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient combined vectorized and scalar Tanh (CUDA)");
}
