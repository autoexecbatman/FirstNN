#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== LibTorch Test ===" << std::endl;
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    
    // Check CUDA availability
    std::cout << "Checking CUDA step by step..." << std::endl;
    std::cout << "1. CUDA compiled in: " << torch::hasCUDA() << std::endl;
    std::cout << "2. CUDA runtime available: " << torch::cuda::is_available() << std::endl;
    std::cout << "3. cuDNN available: " << torch::cuda::cudnn_is_available() << std::endl;
    
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA available: " << std::boolalpha << cuda_available << std::endl;
    
    if (cuda_available) {
        std::cout << "CUDA devices: " << torch::cuda::device_count() << std::endl;
        // Remove current_device() - not available in this API
    }
    
    // Create tensors
    std::cout << "\n=== Tensor Tests ===" << std::endl;
    
    // CPU tensor
    auto cpu_tensor = torch::rand({3, 4});
    std::cout << "CPU tensor (3x4):" << std::endl << cpu_tensor << std::endl;
    
    // GPU tensor if available
    if (cuda_available) {
        auto gpu_tensor = torch::rand({3, 4}).cuda();
        std::cout << "\nGPU tensor (3x4):" << std::endl << gpu_tensor << std::endl;
        
        // Simple operation
        auto result = gpu_tensor * 2.0;
        std::cout << "\nGPU tensor * 2:" << std::endl << result << std::endl;
    }
    
    // Simple neural network test
    std::cout << "\n=== Simple NN Test ===" << std::endl;
    torch::nn::Linear linear(4, 2);
    auto input = torch::rand({1, 4});
    auto output = linear(input);
    std::cout << "Input (1x4): " << input << std::endl;
    std::cout << "Output (1x2): " << output << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
