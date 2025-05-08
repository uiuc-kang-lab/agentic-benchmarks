#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace py = pybind11;

// Utility to parse integer or sequence of integers from Python
inline std::vector<int64_t> parseIntArrayRef(const py::object& obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

// Custom CUDA kernel for transposed convolution (assumes groups==1 and square kernel)
// Processes a chunk of the batch starting at batch_offset with current chunk size equal to gridDim.z
__global__
void conv_transpose2d_kernel(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int batch_offset,
    int C_in, int H_in, int W_in,
    int C_out,
    int kernel_size,
    int stride,
    int padding,
    int H_out, int W_out
) {
    int b = blockIdx.z; // index within the current chunk
    int n = batch_offset + b; // global batch index
    int oc = blockIdx.y * blockDim.y + threadIdx.y;  // output channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // flattened spatial index

    if (oc < C_out && idx < H_out * W_out) {
        int oh = idx / W_out;
        int ow = idx % W_out;
        float value = (bias ? bias[oc] : 0.0f);

        for (int ic = 0; ic < C_in; ic++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int in_h = oh + padding - i;
                    int in_w = ow + padding - j;
                    if ((in_h % stride) == 0 && (in_w % stride) == 0) {
                        int h_in = in_h / stride;
                        int w_in = in_w / stride;
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int input_index  = ((n * C_in + ic) * H_in + h_in) * W_in + w_in;
                            int weight_index = ((ic * C_out + oc) * kernel_size + i) * kernel_size + j;
                            value += input[input_index] * weight[weight_index];
                        }
                    }
                }
            }
        }
        int output_index = ((n * C_out + oc) * H_out + oh) * W_out + ow;
        output[output_index] = value;
    }
}


// Forward function implementing asynchronous overlapping between memory transfers and kernel execution
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0), // not used in our custom kernel
    int64_t groups = 1                    // assumes groups==1
) {
    // Parse the stride and padding arguments
    int stride_val = static_cast<int>(parseIntArrayRef(stride)[0]);
    int padding_val = static_cast<int>(parseIntArrayRef(padding)[0]);

    // Get dimensions
    // Input tensor expected shape: [N, C_in, H_in, W_in]
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    // Weight tensor expected shape: [C_in, C_out, kernel_size, kernel_size] (square kernel assumed)
    int C_out = weight.size(1);
    int kernel_size = weight.size(2);

    // Compute output spatial dimensions
    int H_out = (H_in - 1) * stride_val + kernel_size - 2 * padding_val;
    int W_out = (W_in - 1) * stride_val + kernel_size - 2 * padding_val;

    // Allocate output tensor on the same device as weight
    auto output = torch::empty({N, C_out, H_out, W_out}, weight.options());

    // We'll process the batch in chunks to overlap memory transfers with kernel execution
    // For demonstration, we use a configurable CHUNK_SIZE. This can be tuned for best performance.
    const int CHUNK_SIZE = 1;

    // Create two CUDA streams for overlapping operations
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Determine if the input is on CPU: if so, we asynchronously copy each chunk to device buffers
    bool input_on_cpu = !input.is_cuda();
    float* d_input_buffers[2] = {nullptr, nullptr};
    size_t chunk_elems = CHUNK_SIZE * C_in * H_in * W_in;
    size_t chunk_bytes = chunk_elems * sizeof(float);
    if (input_on_cpu) {
        cudaMalloc(&d_input_buffers[0], chunk_bytes);
        cudaMalloc(&d_input_buffers[1], chunk_bytes);
    }

    // Set up CUDA kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((H_out * W_out + blockDim.x - 1) / blockDim.x,
                 (C_out + blockDim.y - 1) / blockDim.y);
    
    // Process the batch in chunks to enable pipelining
    for (int n0 = 0; n0 < N; n0 += CHUNK_SIZE) {
        int current_chunk = std::min(CHUNK_SIZE, N - n0);
        // Set grid.z to process the current chunk
        dim3 grid = gridDim;
        grid.z = current_chunk;
        
        // Alternate between the two streams and their associated device buffers
        int stream_idx = (n0 / CHUNK_SIZE) % 2;
        cudaStream_t stream = (stream_idx == 0) ? stream0 : stream1;
        
        // Determine pointer for input data
        const float* input_ptr = nullptr;
        if (input_on_cpu) {
            // Asynchronously transfer the input chunk from CPU to GPU
            auto* cpu_ptr = input.data_ptr<float>();
            cudaMemcpyAsync(d_input_buffers[stream_idx],
                            cpu_ptr + n0 * C_in * H_in * W_in,
                            current_chunk * C_in * H_in * W_in * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            input_ptr = d_input_buffers[stream_idx];
        } else {
            // Input is already on GPU, adjust pointer for the current batch offset
            input_ptr = input.data_ptr<float>() + n0 * C_in * H_in * W_in;
        }
        
        // Launch the CUDA kernel on the selected stream for the current batch chunk
        conv_transpose2d_kernel<<<grid, blockDim, 0, stream>>>(
            input_ptr,
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            n0,  // batch offset
            C_in, H_in, W_in,
            C_out,
            kernel_size,
            stride_val,
            padding_val,
            H_out, W_out
        );
    }

    // Synchronize both streams to ensure all async copy and kernel operations are complete
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    // Free temporary buffers and destroy streams
    if (input_on_cpu) {
        cudaFree(d_input_buffers[0]);
        cudaFree(d_input_buffers[1]);
    }
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Async Overlap ConvTranspose2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
