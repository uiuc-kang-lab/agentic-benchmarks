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
__global__ void tanh_kernel_tiled(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int size) {

    using traits = vectorized_traits<scalar_t>;
    constexpr int vec_width = traits::width;
    // Each tile holds blockDim.x * vec_width scalars
    int tile_size = blockDim.x * vec_width;
    extern __shared__ scalar_t tile[];

    int num_tiles = (size + tile_size - 1) / tile_size;

    // Loop over all tiles assigned to this block
    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        int tile_start = tile_idx * tile_size;
        int tid = threadIdx.x;
        int index = tile_start + tid * vec_width;

        // Load data from global memory into shared memory using vectorized loads when possible
        if (index + vec_width - 1 < size) {
            using vec_t = typename traits::vec_t;
            vec_t temp = reinterpret_cast<const vec_t*>(input)[index / vec_width];
            if constexpr (std::is_same<scalar_t, float>::value) {
                tile[tid * vec_width + 0] = temp.x;
                tile[tid * vec_width + 1] = temp.y;
                tile[tid * vec_width + 2] = temp.z;
                tile[tid * vec_width + 3] = temp.w;
            } else {
                tile[tid * vec_width + 0] = temp.x;
                tile[tid * vec_width + 1] = temp.y;
            }
        } else {
            // Handle boundary conditions
            for (int i = 0; i < vec_width; i++) {
                int idx = index + i;
                if (idx < size) {
                    tile[tid * vec_width + i] = input[idx];
                }
            }
        }

        __syncthreads();

        // Process the tile in shared memory
        if (index < size) {
            using vec_t = typename traits::vec_t;
            vec_t vec_in, vec_out;
            if constexpr (std::is_same<scalar_t, float>::value) {
                vec_in.x = tile[tid * vec_width + 0];
                vec_in.y = tile[tid * vec_width + 1];
                vec_in.z = tile[tid * vec_width + 2];
                vec_in.w = tile[tid * vec_width + 3];
            } else {
                vec_in.x = tile[tid * vec_width + 0];
                vec_in.y = tile[tid * vec_width + 1];
            }
            traits::apply(vec_in, vec_out);
            if constexpr (std::is_same<scalar_t, float>::value) {
                tile[tid * vec_width + 0] = vec_out.x;
                tile[tid * vec_width + 1] = vec_out.y;
                tile[tid * vec_width + 2] = vec_out.z;
                tile[tid * vec_width + 3] = vec_out.w;
            } else {
                tile[tid * vec_width + 0] = vec_out.x;
                tile[tid * vec_width + 1] = vec_out.y;
            }
        }

        __syncthreads();

        // Write the processed tile back to global memory
        if (index + vec_width - 1 < size) {
            using vec_t = typename traits::vec_t;
            vec_t temp;
            if constexpr (std::is_same<scalar_t, float>::value) {
                temp.x = tile[tid * vec_width + 0];
                temp.y = tile[tid * vec_width + 1];
                temp.z = tile[tid * vec_width + 2];
                temp.w = tile[tid * vec_width + 3];
            } else {
                temp.x = tile[tid * vec_width + 0];
                temp.y = tile[tid * vec_width + 1];
            }
            reinterpret_cast<vec_t*>(output)[index / vec_width] = temp;
        } else {
            // Handle boundaries during store
            for (int i = 0; i < vec_width; i++) {
                int idx = index + i;
                if (idx < size) {
                    output[idx] = tile[tid * vec_width + i];
                }
            }
        }

        __syncthreads();
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
