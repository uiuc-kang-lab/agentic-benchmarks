#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <algorithm>

// Kernel that computes a chunk (range of rows) of the upper triangular matmul
__global__ void upper_triangular_matmul_chunk_kernel(const float* __restrict__ A,
                                                       const float* __restrict__ B,
                                                       float* __restrict__ C,
                                                       int N,
                                                       int row_start,
                                                       int row_end) {
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = row_start + local_row;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row >= row_end || global_row >= N || col >= N) return;

if (global_row > col) {
    C[global_row * N + col] = 0.0f;
    return;
}

float sum = 0.0f;
for (int k = global_row; k <= col; k++) {
    sum += A[global_row * N + k] * B[k * N + col];
}
C[global_row * N + col] = sum;
}

// Host function using multiple CUDA streams to overlap computation with asynchronous memory transfers
// The computation is split into chunks by rows. For each chunk, after the kernel computes the results
// into a temporary device buffer, an asynchronous device-to-host copy is launched to a pinned memory buffer.
// Finally, these pinned results are copied back to a final device tensor.
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    // Allocate a temporary device buffer for the computed results
    auto C_temp = torch::empty({N, N}, A.options());
    // Allocate final device output tensor
    auto C_final = torch::empty({N, N}, A.options());

    // Allocate host pinned memory as staging area
    float* C_pinned;
    cudaMallocHost(&C_pinned, N * N * sizeof(float));

    // Determine number of streams and rows per chunk
    int numStreams = 4;
    int rowsPerStream = (N + numStreams - 1) / numStreams;
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threads(32, 32);
    // Launch a kernel for each chunk (row range) on its own stream
    for (int i = 0; i < numStreams; i++) {
        int row_start = i * rowsPerStream;
        int row_end = std::min(row_start + rowsPerStream, N);
        if (row_start >= N) break;

        // Compute grid dimensions: full width (N cols) and only the rows in this chunk
        dim3 blocks((N + threads.x - 1) / threads.x,
                    (rowsPerStream + threads.y - 1) / threads.y);

        upper_triangular_matmul_chunk_kernel<<<blocks, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C_temp.data_ptr<float>(),
            N,
            row_start,
            row_end
        );

        // After computing this chunk, asynchronously copy the results from device to pinned host memory
        size_t chunk_bytes = (row_end - row_start) * N * sizeof(float);
        cudaMemcpyAsync(C_pinned + row_start * N,
                        C_temp.data_ptr<float>() + row_start * N,
                        chunk_bytes,
                        cudaMemcpyDeviceToHost,
                        streams[i]);
    }

    // Synchronize all streams and clean up
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Asynchronously copy the pipelined pinned data back to a final device tensor
    cudaMemcpy(C_final.data_ptr<float>(), C_pinned, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(C_pinned);

    return C_final;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Pipelined upper triangular matrix multiplication with overlapped memory transfers");
}
