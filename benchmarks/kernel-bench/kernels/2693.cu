#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel applies LeakyReLU in a vectorized manner and optionally counts negative activations
// using atomicAdd. Atomic operations are used only for updating the negative count to handle
// race conditions on that shared counter, while the element-wise computation writes independently
// to global memory, avoiding unnecessary atomic operations.
__global__ void leaky_relu_vectorized_atomic_kernel(const float* __restrict__ x,
                                                     float* __restrict__ y,
                                                     float negative_slope,
                                                     int n,
                                                     int* negative_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process full groups of 4 floats with vectorized loads
    int vec_n = n / 4;  // number of full float4 groups
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);
    for (int i = idx; i < vec_n; i += stride) {
        float4 in_val = x_vec[i];
        float4 out_val;

        out_val.x = (in_val.x > 0.0f) ? in_val.x : in_val.x * negative_slope;
        if (negative_count != nullptr && in_val.x <= 0.0f) {
            atomicAdd(negative_count, 1);
        }

        out_val.y = (in_val.y > 0.0f) ? in_val.y : in_val.y * negative_slope;
        if (negative_count != nullptr && in_val.y <= 0.0f) {
            atomicAdd(negative_count, 1);
        }

        out_val.z = (in_val.z > 0.0f) ? in_val.z : in_val.z * negative_slope;
        if (negative_count != nullptr && in_val.z <= 0.0f) {
            atomicAdd(negative_count, 1);
        }

        out_val.w = (in_val.w > 0.0f) ? in_val.w : in_val.w * negative_slope;
        if (negative_count != nullptr && in_val.w <= 0.0f) {
            atomicAdd(negative_count, 1);
        }

        y_vec[i] = out_val;
    }

    // Process remaining elements not divisible by 4
    int offset = vec_n * 4;
    for (int i = offset + idx; i < n; i += stride) {
        float val = x[i];
        if (negative_count != nullptr && val <= 0.0f) {
            atomicAdd(negative_count, 1);
        }
        y[i] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Forward function: applies LeakyReLU and optionally accumulates the count of negative activations
// via an atomic update. If the negative_count tensor is provided, it must be a CUDA tensor with one element.

torch::Tensor leaky_relu_forward_atomic(torch::Tensor x, float negative_slope,
                                          c10::optional<torch::Tensor> neg_count_tensor_opt) {
    CHECK_INPUT(x);
    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    // Compute the number of vectorized elements (each handling 4 floats)
    int vec_elements = (n + 3) / 4; // ceiling division
    int blocks = (vec_elements + threads - 1) / threads;
    blocks = blocks > 0 ? blocks : 1;

    int* neg_count_ptr = nullptr;
    if (neg_count_tensor_opt.has_value()) {
        auto neg_tensor = neg_count_tensor_opt.value();
        CHECK_INPUT(neg_tensor);
        TORCH_CHECK(neg_tensor.numel() == 1, "negative count tensor must have exactly one element");
        neg_tensor.zero_();
        neg_count_ptr = neg_tensor.data_ptr<int>();
    }

    leaky_relu_vectorized_atomic_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), negative_slope, n, neg_count_ptr);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_atomic, 
          "LeakyReLU forward with vectorized load and atomic negative count (CUDA)",
          py::arg("x"), py::arg("negative_slope"), py::arg("negative_count") = py::none());
}
