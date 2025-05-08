// Combined CUDA kernel: stream_modular
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the contribution from a single kernel element
// Marked inline for performance
__device__ __forceinline__ float compute_contribution(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int x_base, int w_base,
    int l_out, int k_w,
    int stride, int padding, int dilation,
    int L_in) {

    int l_in_nom = l_out + padding - k_w * dilation;
    // Check if this kernel position aligns with stride
    if (l_in_nom % stride != 0) return 0.0f;
    int l_in = l_in_nom / stride;
    // Validate l_in against input length
    if (l_in < 0 || l_in >= L_in) return 0.0f;
    float x_val = x[x_base + l_in];
    float w_val = weight[w_base + k_w];
    return x_val * w_val;
}

// New kernel that integrates grid-stride loop with stream-based batch pipelining
__global__ void conv_transpose1d_kernel_stream_modular(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    int n_start, int n_end) {  // Process examples in batch range [n_start, n_end)

    int chunk_N = n_end - n_start;  // Number of examples in this stream/chunk
    int total_elements = chunk_N * C_out * L_out;

    // Use grid-stride loop for flexible workload distribution
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Decode the flattened index into (n, c_out, l_out) within current chunk
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n_local = idx / (L_out * C_out);
        int n = n_start + n_local;  // Actual batch index

        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        // Loop over input channels and kernel width
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_base = n * C_in * L_in + c_in * L_in;
            int w_base = c_in * C_out * K_w + c_out * K_w;
            for (int k_w = 0; k_w < K_w; k_w++) {
                sum += compute_contribution(x, weight, x_base, w_base, l_out, k_w, stride, padding, dilation, L_in);
            }
        }
        
        y[n * C_out * L_out + c_out * L_out + l_out] = sum;
    }
}

// Host function: splits the batch into chunks, each processed in its own CUDA stream
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    // Ensure tensors are contiguous and on CUDA
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    const float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    // Extract dimensions
    int N = static_cast<int>(x.size(0));
    int C_in = static_cast<int>(x.size(1));
    int L_in = static_cast<int>(x.size(2));
    int C_out = static_cast<int>(weight.size(1));
    int K_w = static_cast<int>(weight.size(2));
    
    // Compute output length for conv_transpose1d
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Kernel launch parameters
    int threads = 256;
    int n_streams = 2;  // Use two streams for overlapping independent work
    int chunk_size = (N + n_streams - 1) / n_streams;

    // Create CUDA streams
    cudaStream_t streams[2];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Launch the kernel on each stream for a batch chunk
    for (int i = 0; i < n_streams; i++) {
        int n_start = i * chunk_size;
        int n_end = n_start + chunk_size;
        if (n_end > N) n_end = N;
        int current_chunk = n_end - n_start;
        if (current_chunk <= 0) break;

        int total_elements = current_chunk * C_out * L_out;
        int blocks = (total_elements + threads - 1) / threads;

        conv_transpose1d_kernel_stream_modular<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, C_in, C_out, L_in, L_out, K_w,
            stride, padding, dilation,
            n_start, n_end);
    }

    // Synchronize and destroy streams
    for (int i = 0; i < n_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Combined Conv Transpose1D forward (CUDA) with stream pipelining and modular design",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
