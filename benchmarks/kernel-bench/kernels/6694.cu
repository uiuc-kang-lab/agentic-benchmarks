#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define warp size for warp-level reduction
#define WARP_SIZE 32

// Kernel: Each block computes one output element (a product reduction over dim_size elements).
// The kernel uses a strided loop for each thread followed by warp-level reduction using shuffle intrinsics.
__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int dim_size, int stride) {
    // Each block computes one output element; blockIdx.x corresponds to the output index
    int out_idx = blockIdx.x; 
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    float partial = 1.0f;
    int block_threads = blockDim.x;

    // Each thread processes elements in a strided loop over the reduction dimension
    for (int i = tid; i < dim_size; i += block_threads) {
        partial *= input[out_idx + i * stride];
    }

    // Intra-warp reduction using shuffle operations to combine partial products
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial *= __shfl_down_sync(0xffffffff, partial, offset);
    }

    // Each warp's lane 0 writes its result into shared memory
    __shared__ float warp_prod[32];  // Assuming blockDim.x <= 1024 (max 32 warps per block)
    if (lane == 0) {
        warp_prod[warp_id] = partial;
    }
    __syncthreads();

    // First warp reduces the partial results from each warp
    if (tid < WARP_SIZE) {
        int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float val = (tid < numWarps) ? warp_prod[tid] : 1.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val *= __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            output[out_idx] = val;
        }
    }
}

// Forward function: Implements overlapping of kernel computation and memory transfers using CUDA streams.
// The output tensor is computed in chunks. For each chunk, the product reduction kernel is launched
// asynchronously on a compute stream and its result is concurrently copied from device to pinned host memory
// using a separate copy stream. Finally, the pinned host result is copied back to a new device tensor.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Determine the output shape by removing the reduced dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);

    // Compute total number of output elements
    int num_elements = 1;
    for (auto s : sizes) {
        num_elements *= s;
    }

    // Allocate device memory for the output (as a 1D tensor for simplicity)
    torch::Tensor output_device = torch::empty({num_elements}, x.options());

    // Define a chunk size to pipeline kernel execution with memory copies
    const int CHUNK_SIZE = 1024;  // Number of output elements per chunk
    int num_chunks = (num_elements + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Allocate pinned host memory for asynchronous copy of the output
    auto host_options = torch::TensorOptions().dtype(x.dtype()).device(torch::kCPU).pinned_memory(true);
    torch::Tensor output_host = torch::empty({num_elements}, host_options);

    int stride = x.stride(dim);
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output_device.data_ptr<float>();

    int threads = 256;
    size_t shared_mem_size = threads * sizeof(float);

    // Create two CUDA streams: one for kernel computation and one for memory copies
    cudaStream_t compute_stream, copy_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);

    // Process the output in chunks
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_start = chunk * CHUNK_SIZE;
        int chunk_size = std::min(CHUNK_SIZE, num_elements - chunk_start);

        // Launch the product reduction kernel for this chunk on the compute stream.
        // Each block computes one output element; adjust the output pointer by chunk_start.
        prod_reduce_kernel<<<chunk_size, threads, shared_mem_size, compute_stream>>>(
            input_ptr, output_ptr + chunk_start, dim_size, stride
        );

        // Asynchronously copy the computed chunk from device memory to pinned host memory on the copy stream
        cudaMemcpyAsync(output_host.data_ptr<float>() + chunk_start,
                        output_ptr + chunk_start,
                        chunk_size * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        copy_stream);
    }

    // Synchronize both streams to ensure all operations have completed
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(copy_stream);

    // Destroy the created CUDA streams
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);

    // Copy the data from the pinned host memory back to a new device tensor.
    // The from_blob call creates a tensor that shares memory with the host buffer, so we clone it to ensure it resides on the GPU.
    torch::Tensor final_output = torch::from_blob(output_host.data_ptr<float>(), {num_elements}, x.options()).clone();
    final_output = final_output.view(sizes);

    return final_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension with overlapped memory transfers using CUDA streams (CUDA)");
}
