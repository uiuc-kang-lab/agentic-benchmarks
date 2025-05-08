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

// Kernel that uses warp-level primitives for optimization
template <typename scalar_t>
__global__ void tanh_kernel_warp_optimized(
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

    // Use warp-level primitives to handle remaining elements
    if (remainder > 0) {
        int lane_id = threadIdx.x % warpSize;
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        int warp_offset = warp_id * warpSize * vec_width;

        for (int i = lane_id; i < remainder; i += warpSize) {
            int index = warp_offset + num_vec * vec_width + i;
            if (index < size) {
                if constexpr (std::is_same<scalar_t, float>::value) {
                    output[index] = tanhf(input[index]);
                } else {
                    output[index] = tanh(input[index]);
                }
            }
        }
    }
}

// Host function to launch the CUDA kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    // Optimal thread count for modern GPUs
    const int threads = 256;
    // Maximum number of blocks for good occupancy without excessive oversubscription
    const int max_blocks = 65535;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_warp_optimized", ([&] {
        constexpr int vec_width = vectorized_traits<scalar_t>::width;
        int num_vec = size / vec_width;
        
        // Calculate optimal number of blocks based on workload and hardware limits
        int min_blocks_needed = (num_vec + threads - 1) / threads;
        int blocks = min(max_blocks, min_blocks_needed);
        
        // Ensure at least one block for small inputs
        blocks = max(1, blocks);
        tanh_kernel_warp_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Tanh forward (CUDA)");
}
