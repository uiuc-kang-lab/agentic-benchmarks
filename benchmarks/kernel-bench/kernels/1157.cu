#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <vector>

// Define maximum number of elements for B in constant memory
// Adjust MAX_CONST_B_SIZE as needed to ensure it fits in hardware limits (e.g., 64KB for float: 16384 floats)
#define MAX_CONST_B_SIZE 16384
#define CHUNK_SIZE 32
#define NUM_STREAMS 4

// Declare constant memory for B for different types
__constant__ float d_B_float[MAX_CONST_B_SIZE];
__constant__ double d_B_double[MAX_CONST_B_SIZE];
__constant__ __half d_B_half[MAX_CONST_B_SIZE];

// Kernel that computes a chunk of the 3D tensor-matrix multiplication using constant memory for B
// A has dimensions [N, M, K] and B (stored in constant memory) has dimensions [K, L]
// Output has dimensions [N, M, L]
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L,
    int chunk_start) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_elems = CHUNK_SIZE * M * L;
    if (idx < chunk_elems) {
        int local_n = idx / (M * L);
        int rem = idx % (M * L);
        int m = rem / L;
        int l = rem % L;
        int global_n = chunk_start + local_n;
        if (global_n < N) {
            scalar_t sum = 0;
            for (int k = 0; k < K; ++k) {
                scalar_t a_val = A[global_n * M * K + m * K + k];
                scalar_t b_val;
                if constexpr (std::is_same<scalar_t, float>::value) {
                    b_val = d_B_float[k * L + l];
                } else if constexpr (std::is_same<scalar_t, double>::value) {
                    b_val = d_B_double[k * L + l];
                } else if constexpr (std::is_same<scalar_t, __half>::value) {
                    b_val = d_B_half[k * L + l];
                }
                sum += a_val * b_val;
            }
            output[global_n * M * L + m * L + l] = sum;
        }
    }
}

// Host function: Copies B into constant memory and launches the kernel in a pipelined manner
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    // Ensure B fits within constant memory
    int total_B_elems = K * L;
    TORCH_CHECK(total_B_elems <= MAX_CONST_B_SIZE, "B tensor size (K*L) exceeds constant memory capacity.");

    // Copy B into constant memory based on its data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(B.scalar_type(), "copy_B_to_constant", ([&] {
        const int total = total_B_elems;
        if (std::is_same<scalar_t, float>::value) {
            cudaMemcpyToSymbol(d_B_float, B.data_ptr<scalar_t>(), total * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
        } else if (std::is_same<scalar_t, double>::value) {
            cudaMemcpyToSymbol(d_B_double, B.data_ptr<scalar_t>(), total * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
        } else if (std::is_same<scalar_t, __half>::value) {
            cudaMemcpyToSymbol(d_B_half, B.data_ptr<scalar_t>(), total * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice);
        }
    }));

    // Create CUDA streams for pipelining
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 1024;
    const int elems_per_chunk = CHUNK_SIZE * M * L;
    int blocks = (elems_per_chunk + threads - 1) / threads;

    // Launch kernel for each chunk along the N dimension
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
            int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
            module_fn_cuda_kernel<scalar_t><<<blocks, threads, 0, streams[stream_idx]>>>(
                A.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M, K, L,
                chunk_start);
        }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// Macros to check input properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface: exposed to Python
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);
    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "module_fn forward using constant memory for B (CUDA)");
}
