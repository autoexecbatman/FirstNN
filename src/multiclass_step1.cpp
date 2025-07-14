#include <torch/torch.h>
#include <iostream>
#include <vector>

// Generate synthetic 3-class data
// Class 0: sum < 1.0
// Class 1: 1.0 <= sum < 2.0  
// Class 2: sum >= 2.0
std::pair<torch::Tensor, torch::Tensor> generate_multiclass_data(int num_samples) {
    auto inputs = torch::rand({num_samples, 3}) * 1.5f;  // Random 0-1.5 for each input
    auto sums = torch::sum(inputs, 1);  // Sum each row
    
    // Create targets based on sum ranges
    auto targets = torch::zeros({num_samples}, torch::kLong);
    auto mask1 = (sums >= 1.0f) & (sums < 2.0f);
    auto mask2 = (sums >= 2.0f);
    
    targets.masked_fill_(mask1, 1);
    targets.masked_fill_(mask2, 2);
    
    return {inputs, targets};
}

int main() {
    std::cout << "=== Multi-Class Neural Network (3 Classes) ===" << std::endl;
    
    // Check CUDA
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    // Generate data
    std::cout << "\nGenerating training data..." << std::endl;
    auto [train_inputs, train_targets] = generate_multiclass_data(2000);
    train_inputs = train_inputs.to(device);
    train_targets = train_targets.to(device);
    
    auto [test_inputs, test_targets] = generate_multiclass_data(500);
    test_inputs = test_inputs.to(device);
    test_targets = test_targets.to(device);
    
    std::cout << "Training data shape: " << train_inputs.sizes() << std::endl;
    std::cout << "Training targets shape: " << train_targets.sizes() << std::endl;
    
    // Debug: Print first few targets to verify range
    //std::cout << "First 10 targets: ";
    //for (int i = 0; i < 10; ++i) {
    //    std::cout << train_targets[i].item<int>() << " ";
    //}
    //std::cout << std::endl;
    
    // Initialize weights and biases for 3-class output
    // Network: 3 inputs -> 8 hidden -> 3 outputs (softmax)
    auto w1 = torch::randn({3, 8}, torch::TensorOptions().device(device).requires_grad(true));
    auto b1 = torch::zeros({8}, torch::TensorOptions().device(device).requires_grad(true));
    auto w2 = torch::randn({8, 3}, torch::TensorOptions().device(device).requires_grad(true));
    auto b2 = torch::zeros({3}, torch::TensorOptions().device(device).requires_grad(true));
    
    float learning_rate = 0.01f;
    
    std::cout << "\nStarting training..." << std::endl;
    
    for (int epoch = 0; epoch < 10000; ++epoch) {
        // Forward pass
        auto z1 = torch::mm(train_inputs, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto output = torch::softmax(z2, 1);  // Softmax for multi-class
        
        // Cross-entropy loss: log_softmax + nll_loss (stable implementation)
        auto log_probs = torch::log_softmax(z2, 1);
        auto loss = torch::nll_loss(log_probs, train_targets);
        
        // Backward pass
        loss.backward();
        
        // Update weights
        {
            torch::NoGradGuard no_grad;
            w1 -= learning_rate * w1.grad();
            b1 -= learning_rate * b1.grad();
            w2 -= learning_rate * w2.grad();
            b2 -= learning_rate * b2.grad();
            
            // Zero gradients
            w1.grad().zero_();
            b1.grad().zero_();
            w2.grad().zero_();
            b2.grad().zero_();
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    // Test accuracy
    {
        torch::NoGradGuard no_grad;
        auto z1 = torch::mm(test_inputs, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto test_output = torch::softmax(z2, 1);
        
        auto predictions = torch::argmax(test_output, 1);
        auto correct = torch::sum(predictions == test_targets).item<float>();
        auto accuracy = correct / test_targets.size(0) * 100.0f;
        
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
    }
    
    // Manual test examples
    std::cout << "\nManual Test Examples:" << std::endl;
    torch::NoGradGuard no_grad;
    
    std::vector<std::vector<float>> test_cases = {
        {0.2f, 0.3f, 0.4f},  // sum = 0.9 -> class 0
        {0.5f, 0.6f, 0.7f},  // sum = 1.8 -> class 1
        {0.8f, 0.9f, 1.0f}   // sum = 2.7 -> class 2
    };
    
    for (const auto& test : test_cases) {
        auto input = torch::tensor(test).unsqueeze(0).to(device);
        auto z1 = torch::mm(input, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto output = torch::softmax(z2, 1);
        
        auto probabilities = output.squeeze();
        auto prediction = torch::argmax(probabilities).item<int>();
        float sum = test[0] + test[1] + test[2];
        
        int expected_class;
        if (sum < 1.0f) expected_class = 0;
        else if (sum < 2.0f) expected_class = 1;
        else expected_class = 2;
        
        std::cout << "Input: [" << test[0] << ", " << test[1] << ", " << test[2] 
                  << "] Sum: " << sum << " -> Predicted class: " << prediction 
                  << " (expected: " << expected_class << ")" << std::endl;
        std::cout << "Probabilities: [" << probabilities[0].item<float>() 
                  << ", " << probabilities[1].item<float>() 
                  << ", " << probabilities[2].item<float>() << "]" << std::endl;
    }
    
    return 0;
}
