#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void adaptive_max_pool3d_kernel(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    // Decode output coordinates
    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Use shared memory for small kernel sizes (<=5)
    const bool use_shared = (kernel_size <= 5);
    extern __shared__ char shared_mem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int64_t max_index = -1;

    if (use_shared) {
        // Shared memory approach for small kernels
        const int tid = threadIdx.x;
        const int pooling_size = kernel_size * kernel_size * kernel_size;
        
        for (int i = tid; i < pooling_size; i += blockDim.x) {
            const int k_d = i / (kernel_size * kernel_size);
            const int rem = i % (kernel_size * kernel_size);
            const int k_h = rem / kernel_size;
            const int k_w = rem % kernel_size;

            const int d_in = d_start + k_d * dilation;
            const int h_in = h_start + k_h * dilation;
            const int w_in = w_start + k_w * dilation;

            if (d_in >= 0 && d_in < input_d &&
                h_in >= 0 && h_in < input_h &&
                w_in >= 0 && w_in < input_w) {
                const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                    h_in * input_w + w_in;
                const scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
            }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
        }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}

        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            const scalar_t other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            const int64_t other_idx = __shfl_down_sync(0xffffffff, max_index, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_index = other_idx;
            }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
        }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
    } else {
        // Direct computation with loop unrolling for large kernels
        #pragma unroll 4
        for (int k_d = 0; k_d < kernel_size; k_d++) {
            const int d_in = d_start + k_d * dilation;
            if (d_in < 0 || d_in >= input_d) continue;

            #pragma unroll 4
            for (int k_h = 0; k_h < kernel_size; k_h++) {
                const int h_in = h_start + k_h * dilation;
                if (h_in < 0 || h_in >= input_h) continue;

                #pragma unroll 4
                for (int k_w = 0; k_w < kernel_size; k_w++) {
                    const int w_in = w_start + k_w * dilation;
                    if (w_in < 0 || w_in >= input_w) continue;

                    const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                        h_in * input_w + w_in;
                    const scalar_t val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_index = input_idx;
                    }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
                }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
            }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
        }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
    }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }

// Host wrapper function for adaptive max pool3d forward
#include <torch/extension.h>

torch::Tensor adaptive_max_pool3d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Get input dimensions
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_d = input.size(2);
    const auto input_h = input.size(3);
    const auto input_w = input.size(4);

    // Compute output dimensions (assuming standard convolution formula)
    const int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate output and indices tensors
    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kInt64));

    // Total number of threads to launch
    const int total_threads = batch_size * channels * output_d * output_h * output_w;
    const int threads = 1024; // block size
    const int blocks = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cuda", ([&] {
        // Launch kernel. Note: dynamic shared memory size is set to 0 here. If needed, adjust accordingly.
        adaptive_max_pool3d_kernel<scalar_t><<<blocks, threads, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    // It is important to check for CUDA errors in a production kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// PyBind module definition to expose the CUDA kernel
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_max_pool3d_forward", &adaptive_max_pool3d_forward, "Adaptive max pool3d forward (CUDA)");
}
}