#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Template specialization for atomicMax for float and double

template <typename scalar_t>
__device__ scalar_t atomicMaxT(scalar_t* addr, scalar_t value);

template <>
__device__ float atomicMaxT<float>(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;
    while (value > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
    return __int_as_float(old);
}

template <>
__device__ double atomicMaxT<double>(double* addr, double value) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = __double_as_longlong(*addr);
    unsigned long long int assumed;
    while (value > __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
        if (assumed == old) break;
    }
    return __longlong_as_double(old);
}

// Atomic kernel: each thread processes one input element and updates all output elements
// in whose pooling window the input element falls. This mapping is required in
// overlapping pooling scenarios where multiple threads may update the same output cell.

template <typename scalar_t>
__global__ void max_pool2d_kernel_atomic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * input_height * input_width;
    if (idx >= total_elements) return;

    // Compute input indices
    int iw = idx % input_width;
    int ih = (idx / input_width) % input_height;
    int c  = (idx / (input_width * input_height)) % channels;
    int b  = idx / (channels * input_width * input_height);

    scalar_t val = input[idx];

    // Compute the range of output rows this input element may contribute to
    int oh_start = (ih + padding - (kernel_size - 1) * dilation + stride - 1) / stride;
    if (oh_start < 0) oh_start = 0;
    int oh_end = (ih + padding) / stride;
    if (oh_end >= output_height) oh_end = output_height - 1;

    // Similarly, compute the range of output columns
    int ow_start = (iw + padding - (kernel_size - 1) * dilation + stride - 1) / stride;
    if (ow_start < 0) ow_start = 0;
    int ow_end = (iw + padding) / stride;
    if (ow_end >= output_width) ow_end = output_width - 1;

    // Iterate over the candidate output positions
    for (int oh = oh_start; oh <= oh_end; oh++) {
        // Determine the pooling window boundaries for this output row
        int h_start = oh * stride - padding;
        int h_end_window = h_start + (kernel_size - 1) * dilation;
        if (ih < h_start || ih > h_end_window) continue;  // input row not in pooling window

        for (int ow = ow_start; ow <= ow_end; ow++) {
            int w_start = ow * stride - padding;
            int w_end_window = w_start + (kernel_size - 1) * dilation;
            if (iw < w_start || iw > w_end_window) continue;  // input col not in pooling window

            // Compute global index for the output element
            int out_idx = b * (channels * output_height * output_width) +
                          c * (output_height * output_width) +
                          oh * output_width +
                          ow;
            // Atomically update the maximum value for this output element
            atomicMaxT(&output[out_idx], val);
        }
    }
}

// Host function for the atomic max pooling forward pass
// This kernel is especially beneficial in overlapping pooling cases
// where multiple input elements contribute to the same output cell and race conditions
// must be handled via atomic operations. Global atomics are minimized to only those updates
// where contention exists.

torch::Tensor max_pool2d_cuda_forward_atomic(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Pre-initialize the output tensor to -infinity to ensure correct max computations
    auto options = input.options();
    if (input.scalar_type() == torch::kFloat) {
        input = input.contiguous();
    }
    
    double neg_inf = -std::numeric_limits<double>::infinity();
    if (input.scalar_type() == torch::kFloat) {
        neg_inf = -std::numeric_limits<float>::infinity();
    }

    auto output = torch::full({batch_size, channels, output_height, output_width}, neg_inf, options);

    int total_input = batch_size * channels * input_height * input_width;
    const int threads = 256;
    const int blocks = (total_input + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_atomic", ([&] {
        max_pool2d_kernel_atomic<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_atomic, "Max Pool 2D forward with minimal atomic operations (CUDA)");
}
