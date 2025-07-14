#include <torch/torch.h>
#include <iostream>
#include <vector>

// Generate training data: predict if sum of 3 numbers > 1.5
std::pair<torch::Tensor, torch::Tensor> generate_data(int batch_size) {
    auto inputs = torch::rand({batch_size, 3});  // 3 random numbers 0-1
    auto sums = torch::sum(inputs, 1);           // Sum each row
    auto targets = (sums > 1.5).to(torch::kFloat); // 1 if sum > 1.5, else 0
    return {inputs, targets};
}

int main() {
    std::cout << "=== Binary Classifier Training (Manual Implementation) ===" << std::endl;
    
    // Check CUDA availability
    std::cout << "CUDA compiled in: " << torch::hasCUDA() << std::endl;
    std::cout << "CUDA runtime available: " << torch::cuda::is_available() << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA devices: " << torch::cuda::device_count() << std::endl;
    }
    
    // Setup
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << device << std::endl;
    
    // Manual network weights (no modules) - create directly on device to avoid non-leaf
    auto w1 = torch::randn({3, 10}, torch::TensorOptions().device(device).requires_grad(true));
    auto b1 = torch::zeros({10}, torch::TensorOptions().device(device).requires_grad(true));
    auto w2 = torch::randn({10, 1}, torch::TensorOptions().device(device).requires_grad(true));
    auto b2 = torch::zeros({1}, torch::TensorOptions().device(device).requires_grad(true));
    
    // Xavier initialization (detach and reattach gradients)
    {
        torch::NoGradGuard no_grad;
        torch::nn::init::xavier_uniform_(w1);
        torch::nn::init::xavier_uniform_(w2);
    }
    w1.requires_grad_(true);
    w2.requires_grad_(true);
    
    // Optimizer parameters
    std::vector<torch::Tensor> parameters = {w1, b1, w2, b2};
    float learning_rate = 0.01;
    
    std::cout << "Training to predict: sum of 3 numbers > 1.5" << std::endl;
    std::cout << "Network: 3 -> 10 -> 1 (manual implementation)" << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < 20000; ++epoch) {
        auto [inputs, targets] = generate_data(100);
        inputs = inputs.to(device);
        targets = targets.to(device);
        
        // Forward pass (manual)
        auto z1 = torch::mm(inputs, w1) + b1;  // Linear layer 1
        auto a1 = torch::relu(z1);             // ReLU activation
        auto z2 = torch::mm(a1, w2) + b2;      // Linear layer 2  
        auto outputs = torch::sigmoid(z2);    // Sigmoid output
        
        // Binary Cross Entropy loss (manual)
        auto loss = -torch::mean(targets * torch::log(outputs.squeeze() + 1e-8) + 
                                (1 - targets) * torch::log(1 - outputs.squeeze() + 1e-8));
        
        // Backward pass
        loss.backward();
        
        // Manual optimizer step (SGD) - use data() to avoid non-leaf warnings
        {
            torch::NoGradGuard no_grad;
            if (w1.grad().defined()) {
                w1.data() -= learning_rate * w1.grad().data();
                w1.grad().zero_();
            }
            if (b1.grad().defined()) {
                b1.data() -= learning_rate * b1.grad().data();
                b1.grad().zero_();
            }
            if (w2.grad().defined()) {
                w2.data() -= learning_rate * w2.grad().data();
                w2.grad().zero_();
            }
            if (b2.grad().defined()) {
                b2.data() -= learning_rate * b2.grad().data();
                b2.grad().zero_();
            }
        }
        
        if ((epoch + 1) % 200 == 0) {
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
    
    // Test with specific examples - more comprehensive test cases
    std::cout << "\n=== Testing ===" << std::endl;
    
    // Test cases
    std::vector<std::vector<float>> test_cases = {
        // Clear negatives (sum < 1.5)
        {0.1, 0.2, 0.3},  // Sum = 0.6
        {0.0, 0.4, 0.5},  // Sum = 0.9
        {0.2, 0.3, 0.4},  // Sum = 0.9
        {0.1, 0.1, 0.8},  // Sum = 1.0
        {0.0, 0.7, 0.7},  // Sum = 1.4
        
        // Boundary cases (sum = 1.5)
        {0.5, 0.5, 0.5},  // Sum = 1.5 (exact boundary)
        {0.3, 0.6, 0.6},  // Sum = 1.5 (exact boundary)
        {0.1, 0.7, 0.7},  // Sum = 1.5 (exact boundary)
        
        // Clear positives (sum > 1.5)
        {0.8, 0.9, 0.7},  // Sum = 2.4
        {0.6, 0.6, 0.6},  // Sum = 1.8
        {0.3, 0.6, 0.8},  // Sum = 1.7
        {0.9, 0.9, 0.9},  // Sum = 2.7
        {0.5, 0.5, 0.8},  // Sum = 1.8
        {1.0, 1.0, 1.0},  // Sum = 3.0 (maximum)
        
        // Edge cases
        {0.0, 0.0, 2.0},  // Sum = 2.0 (one large)
        {1.5, 0.0, 0.0},  // Sum = 1.5 (boundary, one large)
        {0.51, 0.51, 0.51}, // Sum = 1.53 (just over)
        {0.49, 0.49, 0.52}  // Sum = 1.50 (at boundary)
    };
    
    torch::NoGradGuard no_grad;
    for (auto& test : test_cases) {
        auto input = torch::tensor(test).unsqueeze(0).to(device);
        
        // Forward pass
        auto z1 = torch::mm(input, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto output = torch::sigmoid(z2);
        
        float prediction = output.item<float>();
        float sum = test[0] + test[1] + test[2];
        bool actual = sum > 1.5;
        std::string result = (prediction > 0.5) == actual ? "PASS" : "FAIL";
        
        std::cout << "Input: [" << test[0] << ", " << test[1] << ", " << test[2] 
                  << "] Sum: " << sum << " -> Prediction: " << prediction 
                  << " (should be " << actual << ") " << result << std::endl;
    }
    
    // Accuracy test
    auto [test_inputs, test_targets] = generate_data(1000);
    test_inputs = test_inputs.to(device);
    test_targets = test_targets.to(device);
    
    // Forward pass for accuracy
    auto z1 = torch::mm(test_inputs, w1) + b1;
    auto a1 = torch::relu(z1);
    auto z2 = torch::mm(a1, w2) + b2;
    auto test_outputs = torch::sigmoid(z2);
    
    auto predictions = (test_outputs.squeeze() > 0.5).to(torch::kFloat);
    auto accuracy = torch::sum(predictions == test_targets).item<float>() / 1000.0f;
    
    std::cout << "\nAccuracy on 1000 test samples: " << accuracy * 100 << "%" << std::endl;
    
    return 0;
}