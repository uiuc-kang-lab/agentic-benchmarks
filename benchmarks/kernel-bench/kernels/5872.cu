#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Implementation of atomicMax for float using bit-level compare-and-swap
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val)
            break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Implementation of atomicMax for double using atomicCAS on unsigned long long integers
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val)
            break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Template wrapper for atomicMax on shared memory
template <typename scalar_t>
__device__ __forceinline__ scalar_t atomicMaxCustom(scalar_t* address, scalar_t val);

template <>
__device__ __forceinline__ float atomicMaxCustom<float>(float* address, float val) {
    return atomicMaxFloat(address, val);
}

template <>
__device__ __forceinline__ double atomicMaxCustom<double>(double* address, double val) {
    return atomicMaxDouble(address, val);
}

// Optimized kernel where each block computes one output element using shared memory reduction
// The pooling window is scanned in parallel by threads in the block and the maximum is updated
// in shared memory using atomic operations (which are fast in shared memory).
template <typename scalar_t>
__global__ void max_pool3d_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    // Each block handles one output element
    const int out_idx = blockIdx.x;
    if (out_idx >= batch_size * channels * output_d * output_h * output_w)
        return;

    // Decode linear index into 5D coordinates: b, c, d_out, h_out, w_out
    int w_out = out_idx % output_w;
    int h_out = (out_idx / output_w) % output_h;
    int d_out = (out_idx / (output_w * output_h)) % output_d;
    int c = (out_idx / (output_w * output_h * output_d)) % channels;
    int b = out_idx / (output_w * output_h * output_d * channels);

    // Compute starting coordinates in input (taking into account stride and padding)
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // Allocate one element in shared memory to hold the block's maximum value
    extern __shared__ char smem[];
    scalar_t* smax = reinterpret_cast<scalar_t*>(smem);
    if (threadIdx.x == 0) {
        smax[0] = -std::numeric_limits<scalar_t>::infinity();
    }
    __syncthreads();

    // Total number of elements in the pooling window
    const int pooling_volume = kernel_size * kernel_size * kernel_size;

    // Each thread processes a subset of the pooling window
    for (int idx = threadIdx.x; idx < pooling_volume; idx += blockDim.x) {
        int k_d = idx / (kernel_size * kernel_size);
        int rem = idx % (kernel_size * kernel_size);
        int k_h = rem / kernel_size;
        int k_w = rem % kernel_size;

        int d_in = d_start + k_d * dilation;
        int h_in = h_start + k_h * dilation;
        int w_in = w_start + k_w * dilation;

        // Check bounds
        if (d_in >= 0 && d_in < input_d &&
            h_in >= 0 && h_in < input_h &&
            w_in >= 0 && w_in < input_w) {
            int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                            h_in * input_w + w_in;
            scalar_t val = input[input_idx];
            // Update the shared memory maximum using an atomic operation in fast shared memory
            atomicMaxCustom(&smax[0], val);
        }
    }
    __syncthreads();

    // Thread 0 writes the final max value (and corresponding index) to global memory
    if (threadIdx.x == 0) {
        output[out_idx] = smax[0];
        if (indices != nullptr) {
            // Re-scan the pooling window to determine the index corresponding to the max value
            scalar_t max_val = smax[0];
            int max_index = -1;
            for (int k_d = 0; k_d < kernel_size; k_d++) {
                int d_in = d_start + k_d * dilation;
                if (d_in < 0 || d_in >= input_d) continue;
                for (int k_h = 0; k_h < kernel_size; k_h++) {
                    int h_in = h_start + k_h * dilation;
                    if (h_in < 0 || h_in >= input_h) continue;
                    for (int k_w = 0; k_w < kernel_size; k_w++) {
                        int w_in = w_start + k_w * dilation;
                        if (w_in < 0 || w_in >= input_w) continue;
                        int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                        h_in * input_w + w_in;
                        if (input[input_idx] == max_val) {
                            max_index = input_idx;
                            goto finish; // break out once found
                        }
                    }
                }
            }
            finish:
            indices[out_idx] = max_index;
        }
    }
}

// Host wrapper function for the optimized max pooling forward kernel
torch::Tensor max_pool3d_cuda_forward_optimized(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {
    
    auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    // Compute output dimensions
    const int output_d = ceil_mode ?
        static_cast<int>(ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)) :
        static_cast<int>(floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1));
    const int output_h = ceil_mode ?
        static_cast<int>(ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)) :
        static_cast<int>(floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1));
    const int output_w = ceil_mode ?
        static_cast<int>(ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)) :
        static_cast<int>(floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1));

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int total_outputs = batch_size * channels * output_d * output_h * output_w;
    // Launch one block per output element. Each block uses a number of threads (e.g., 256) to parallelize the reduction
    const int threads = 256;
    const int blocks = total_outputs;
    // Shared memory size is sizeof(scalar_t) (one element per block)
    size_t shared_mem_size = input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_optimized", ([&] {
        max_pool3d_forward_kernel_optimized<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward_optimized, "Optimized Max Pool 3D forward (CUDA)");
}
