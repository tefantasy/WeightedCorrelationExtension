#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor weighted_corr_cuda_forward(
        torch::Tensor input1, torch::Tensor input2, 
        torch::Tensor weights,
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups);

std::vector<torch::Tensor> weighted_corr_cuda_backward(
        torch::Tensor input1, torch::Tensor input2, 
        torch::Tensor weights, torch::Tensor grad_output, 
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups);



// only support CUDA mode up to now

torch::Tensor weighted_corr_forward(
        torch::Tensor input1, torch::Tensor input2, 
        torch::Tensor weights,
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups) {
    CHECK_CUDA(input1);
    CHECK_CUDA(input2);
    CHECK_CUDA(weights);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input1));

    return weighted_corr_cuda_forward(
        input1, input2, weights, 
        in_channel, filter_size, dilation, num_groups
    );
}

std::vector<torch::Tensor> weighted_corr_backward(
        torch::Tensor input1, torch::Tensor input2, 
        torch::Tensor weights, torch::Tensor grad_output, 
        const int in_channel, const int filter_size,
        const int dilation, const int num_groups) {
    CHECK_CUDA(input1);
    CHECK_CUDA(input2);
    CHECK_CUDA(weights);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input1));

    return weighted_corr_cuda_backward(
        input1, input2, weights, grad_output, 
        in_channel, filter_size, dilation, num_groups
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &weighted_corr_forward, "Weighed correlation operation forward (CUDA)");
  m.def("backward", &weighted_corr_backward, "Weighed correlation operation backward (CUDA)");
}
