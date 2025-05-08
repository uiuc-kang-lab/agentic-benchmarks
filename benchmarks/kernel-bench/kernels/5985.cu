#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel minimizes warp divergence by using branchless computations for handling boundaries
// It precomputes the valid range indices for the kernel window and uses a mask to zero out contributions
// from out-of-bound indices without introducing divergent branches in the inner loop.

__global__ void warp_uniform_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    // Flattened thread index covering all output elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * output_length;
    if (idx >= total_elements) return;

    // Compute output coordinates: o (position), channel, and batch index
    int o = idx % output_length;
    int channel = (idx / output_length) % in_channels;
    int batch = idx / (output_length * in_channels);

    // Compute pointer offset for current (batch, channel)
    int input_offset = batch * in_channels * input_length + channel * input_length;

    // Compute the starting index in input corresponding to the beginning of the kernel window
    int start_idx = o * stride - padding;

    // Compute the valid range of kernel indices [k_start, k_end) such that (start_idx + k) is in bounds
    // Using branchless expressions here via ternary operators (which most compilers compile as predicated moves)
    int k_start = (start_idx < 0) ? -start_idx : 0;
    int valid_input = input_length - start_idx;   
    valid_input = (valid_input < 0) ? 0 : valid_input;
    int k_end = (kernel_size < valid_input) ? kernel_size : valid_input;

    float sum = 0.0f;
    // Iterate over the entire kernel window uniformly
    // Every thread loops exactly 'kernel_size' iterations to avoid divergence
    for (int k = 0; k < kernel_size; ++k) {
        // Compute the global input index for this kernel element
        int idx_k = start_idx + k;
        // Branchless clamp: if idx_k is out of bounds, clamp it to a safe index (0 or input_length-1).
        // The contribution of these accesses will be nullified by the mask below.
        idx_k = (idx_k < 0) ? 0 : (idx_k >= input_length ? input_length - 1 : idx_k);

        // Compute a mask that is 1 if k is within the valid [k_start, k_end) range, 0 otherwise.
        // This avoids a divergent if() inside the loop.
        int valid = ((unsigned)(k - k_start) < (unsigned)(k_end - k_start)) ? 1 : 0;
        
        // Only valid indices contribute; invalid ones multiply the loaded value by 0
        float val = input[input_offset + idx_k] * valid;
        sum += val;
    }

    // Normalize the sum by kernel_size to get the average
    output[idx] = sum / kernel_size;
}

torch::Tensor warp_uniform_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());
    int total_elements = batch_size * in_channels * output_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    warp_uniform_avg_pool1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_uniform_avg_pool1d_forward, "Uniform control flow 1D Average Pooling forward (CUDA)");
}
