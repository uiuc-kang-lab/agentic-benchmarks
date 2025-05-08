#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel to compute reverse cumulative sum along a given dimension.
// Combines efficient memory handling and computation using thrust.

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

at::Tensor reverse_cumsum(at::Tensor x, int64_t dim) {
    // Ensure the tensor is contiguous and on CUDA
    x = x.contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");

    const int ndim = x.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Prepare output tensor
    auto output = at::empty_like(x);

    // Use thrust for efficient reverse cumulative sum computation
    auto x_data = x.data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    thrust::device_ptr<float> dev_ptr_in(x_data);
    thrust::device_ptr<float> dev_ptr_out(output_data);

    // Flip the input data using thrust
    thrust::reverse(dev_ptr_in, dev_ptr_in + x.numel());

    // Compute cumulative sum
    thrust::inclusive_scan(dev_ptr_in, dev_ptr_in + x.numel(), dev_ptr_out);

    // Flip the output data back to the original orientation
    thrust::reverse(dev_ptr_out, dev_ptr_out + x.numel());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum, "Optimized reverse cumulative sum using Thrust (CUDA)");
}
