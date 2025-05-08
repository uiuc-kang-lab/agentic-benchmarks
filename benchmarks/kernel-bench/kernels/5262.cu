#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel: processes a slice of the batch dimension
__global__ void max_pool1d_kernel_stream(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length) {

    // b is the relative batch index within this slice
    int b = blockIdx.z;
    int c = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_length) return;

    int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    for (int k = 0; k < kernel_size; k++) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = input[b * num_channels * input_length + c * input_length + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (indices != nullptr) {
        indices[out_idx] = max_idx;
    }
}

// Host function that partitions the batch across multiple streams to overlap kernel execution with memory operations
torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
                                 torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    // Define kernel launch configuration for output elements
    const int threads_x = 256;
    const dim3 threads(threads_x);
    const dim3 grid((output_length + threads_x - 1) / threads_x, num_channels);

    // Partition the batch dimension across several streams to overlap compute with any memory ops
    int num_streams = (batch_size < 4) ? batch_size : 4;  // use up to 4 streams
    int slice_size = (batch_size + num_streams - 1) / num_streams;

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Launch the kernel asynchronously on each stream for its batch slice
    for (int i = 0; i < num_streams; i++) {
        int start_batch = i * slice_size;
        if (start_batch >= batch_size) break;
        int current_slice = std::min(slice_size, batch_size - start_batch);

        const float* input_ptr = x.data_ptr<float>() + start_batch * num_channels * input_length;
        float* output_ptr = output.data_ptr<float>() + start_batch * num_channels * output_length;
        int64_t* indices_ptr = return_indices ? (indices.data_ptr<int64_t>() + start_batch * num_channels * output_length) : nullptr;

        dim3 grid_slice = dim3(grid.x, grid.y, current_slice);

        max_pool1d_kernel_stream<<<grid_slice, threads, 0, streams[i]>>>(
            input_ptr,
            output_ptr,
            indices_ptr,
            num_channels,
            input_length,
            kernel_size,
            stride,
            padding,
            dilation,
            output_length
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with overlapping streams (CUDA)");
}
