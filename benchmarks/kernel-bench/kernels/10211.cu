#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

__global__ void forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W,
    int batch_offset
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    
    if (index >= total_elements) return;
    
    // Decompose linear index into 4D tensor coordinates
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = (index / (W * H * OC)) + batch_offset;

    float sum = 0.0f;
    
    // 1x1 convolution equivalent to matmul over channels
    #pragma unroll
    for (int ic = 0; ic < IC; ++ic) {
        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        const int w_offset = oc * IC + ic;
        sum += x[x_offset] * weight[w_offset];
    }
    
    // Handle optional bias
    output[index] = bias ? sum + bias[oc] : sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    // Input validation
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate batch size per stream
    const int batch_per_stream = (B + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;

    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int stream_batch_start = i * batch_per_stream;
        int stream_batch_size = std::min(batch_per_stream, B - stream_batch_start);
        
        if (stream_batch_size <= 0) continue;

        const int elements = stream_batch_size * OC * H * W;
        const int blocks = (elements + threads - 1) / threads;

        forward_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_ptr, w_ptr, b_ptr, out_ptr,
            stream_batch_size, IC, OC, H, W,
            stream_batch_start
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}