#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of the input using a grid-stride loop
template <typename scalar_t>
__global__ void relu_kernel_chunk(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // Use 4-element vector types for coalesced memory access when possible
    using Vec4 = typename std::conditional<std::is_same<scalar_t, float>::value,
        float4, double4>::type;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vector processing for aligned data
    const int vec_size = 4;
    const int64_t vec_elements = size / vec_size;
    const Vec4* input_vec = reinterpret_cast<const Vec4*>(input);
    Vec4* output_vec = reinterpret_cast<Vec4*>(output);
    
    // Process 4 elements at a time
    for (int64_t i = idx; i < vec_elements; i += stride) {
        Vec4 in = input_vec[i];
        if constexpr (std::is_same<scalar_t, float>::value) {
            in.x = (in.x > 0) ? in.x : 0.0f;
            in.y = (in.y > 0) ? in.y : 0.0f;
            in.z = (in.z > 0) ? in.z : 0.0f;
            in.w = (in.w > 0) ? in.w : 0.0f;
        } else {
            in.x = (in.x > 0) ? in.x : 0.0;
            in.y = (in.y > 0) ? in.y : 0.0;
            in.z = (in.z > 0) ? in.z : 0.0;
            in.w = (in.w > 0) ? in.w : 0.0;
        }
        output_vec[i] = in;
    }
    
    // Handle remaining elements
    const int64_t remaining_start = vec_elements * vec_size;
    for (int64_t i = remaining_start + idx; i < size; i += stride) {
        scalar_t x = input[i];
        output[i] = (x > 0) ? x : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function that splits the work into chunks processed on separate CUDA streams
torch::Tensor forward(torch::Tensor input) {
    // Get total number of elements
    const int64_t numel = input.numel();
    // Allocate output tensor (uninitialized memory)
    auto output = torch::empty_like(input);

    const int threads = 256;
    // Choose number of streams; for small tensors use a single stream
    int num_streams = 4;
    if (numel < 4096) {
        num_streams = 1;
    }
    // Determine chunk size per stream
    const int64_t chunk_size = (numel + num_streams - 1) / num_streams;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Launch kernel for each chunk on its own stream
    for (int i = 0; i < num_streams; i++) {
        int64_t offset = i * chunk_size;
        if (offset >= numel) break;
        int64_t current_chunk = std::min(chunk_size, numel - offset);
        int blocks = (current_chunk + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_chunk", ([&] {
            relu_kernel_chunk<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                output.data_ptr<scalar_t>() + offset,
                input.data_ptr<scalar_t>() + offset,
                current_chunk);
        }));
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed ReLU forward (CUDA)");
}
