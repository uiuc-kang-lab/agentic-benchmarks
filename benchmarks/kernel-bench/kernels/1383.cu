#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel to compute a chunk of the diagonal matrix multiplication
__global__ void diag_matmul_stream_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t row_offset,
    const int64_t chunk_rows,
    const int64_t M
) {
    int row_in_chunk = blockIdx.x; // one block per row in the chunk
    int row = row_offset + row_in_chunk;
    if (row_in_chunk >= chunk_rows) return;
    
    // Load the diagonal element from A
    float a_val = A[row];

    int thread_id = threadIdx.x;
    int stride = blockDim.x;

    int vec_limit = M / 4;
    const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
    float4* C_vec = reinterpret_cast<float4*>(C + row * M);

    // Process vectorized part
    while (thread_id < vec_limit) {
        float4 b_val = B_vec[thread_id];
        float4 c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        C_vec[thread_id] = c_val;
        thread_id += stride;
    }

    // Process remaining tail elements
    int offset = (M / 4) * 4;
    thread_id = threadIdx.x;
    while (offset + thread_id < M) {
        C[row * M + offset + thread_id] = a_val * B[row * M + offset + thread_id];
        thread_id += stride;
    }
}

// Forward function that overlaps computation with memory transfers using CUDA streams
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    size_t total_bytes = N * M * sizeof(float);

    // Allocate device output tensor (results remain on the device initially)
    auto C_dev = torch::empty({N, M}, B.options());

    // We'll use multiple CUDA streams to overlap kernel execution and memory transfers
    int num_streams = 4;
    // Compute the number of rows each stream will process
    int chunk_size = (N + num_streams - 1) / num_streams;

    // Allocate pinned host memory for the final output
    float* C_host_ptr = nullptr;
    cudaMallocHost((void**)&C_host_ptr, total_bytes);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threads = 256; // threads per block
    for (int i = 0; i < num_streams; i++) {
        int row_offset = i * chunk_size;
        if (row_offset >= N) break;
        int current_chunk = std::min(chunk_size, (int)(N - row_offset));
        
        // Launch the kernel for this chunk asynchronously on stream[i]
        diag_matmul_stream_kernel<<<current_chunk, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C_dev.data_ptr<float>(),
            row_offset,
            current_chunk,
            M
        );
        
        // Asynchronously copy the computed chunk from device to pinned host memory
        size_t chunk_bytes = current_chunk * M * sizeof(float);
        cudaMemcpyAsync(
            C_host_ptr + row_offset * M,              // destination in host memory
            C_dev.data_ptr<float>() + row_offset * M,   // source in device memory
            chunk_bytes,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // Synchronize all streams to ensure completion
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Create a CPU tensor from the pinned host memory. Clone to ensure the tensor owns its data.
    auto options = torch::TensorOptions().dtype(B.dtype()).device(torch::kCPU);
    at::Tensor C = torch::from_blob(C_host_ptr, {N, M}, options).clone();
    cudaFreeHost(C_host_ptr);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with overlapped memory transfers using CUDA streams");
}
