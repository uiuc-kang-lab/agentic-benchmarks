#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that processes a chunk of the batch
__global__ void forward_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    // Decompose linear index into 4D tensor coordinates (for the current batch chunk)
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        int weight_offset = oc * IC + ic;
        sum += x[x_offset] * weight[weight_offset];
    }

    if (bias) {
        output[index] = sum + bias[oc];
    } else {
        output[index] = sum;
    }
}

// Forward function using CUDA streams to overlap kernel execution and memory operations
torch::Tensor forward_cuda_stream(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) {
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Decide on the number of streams (using 2 streams if batch dimension > 1)
    int num_streams = (B > 1) ? 2 : 1;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Partition the batch dimension among streams
    int batch_offset = 0;
    for (int s = 0; s < num_streams; s++) {
        int chunk = B / num_streams;  // base number of batches per stream
        if (s < (B % num_streams)) {
            chunk++;
        }
        if (chunk == 0) continue;

        // Compute the number of elements for this batch chunk
        int chunk_elements = chunk * OC * H * W;
        int threads = 256;
        int blocks = (chunk_elements + threads - 1) / threads;

        // Compute pointer offsets for the current batch chunk
        const float* x_chunk_ptr = x_ptr + batch_offset * IC * H * W;
        float* output_chunk_ptr = output_ptr + batch_offset * OC * H * W;

        // Launch the kernel on the current stream
        forward_kernel<<<blocks, threads, 0, streams[s]>>>(
            x_chunk_ptr, weight_ptr, bias_ptr, output_chunk_ptr,
            chunk, IC, OC, H, W
        );

        batch_offset += chunk;
    }

    // Synchronize and destroy streams
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda_stream, "Pointwise 2D convolution forward with streams (CUDA)");
}
