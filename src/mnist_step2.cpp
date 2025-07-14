#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>

// Simple MNIST data loader (manual implementation)
class SimpleMNIST {
private:
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    
public:
    bool load_data(const std::string& images_file, const std::string& labels_file, int max_samples = 1000) {
        // For now, create synthetic MNIST-like data
        // TODO: Replace with actual MNIST file loading
        std::cout << "Generating synthetic MNIST-like data..." << std::endl;
        
        for (int i = 0; i < max_samples; ++i) {
            std::vector<float> image(784); // 28x28 = 784 pixels
            
            // Create simple pattern: vertical lines for digit simulation
            int digit = i % 10;
            for (int pixel = 0; pixel < 784; ++pixel) {
                int row = pixel / 28;
                int col = pixel % 28;
                
                // Simple pattern based on digit
                if (digit == 1) {
                    image[pixel] = (col >= 12 && col <= 15) ? 1.0f : 0.0f;
                } else if (digit == 0) {
                    image[pixel] = ((col >= 8 && col <= 10) || (col >= 17 && col <= 19)) && 
                                  (row >= 8 && row <= 19) ? 1.0f : 0.0f;
                } else {
                    // Random pattern for other digits
                    image[pixel] = (rand() % 100 < (digit * 5 + 10)) ? 1.0f : 0.0f;
                }
            }
            
            images.push_back(image);
            labels.push_back(digit);
        }
        
        std::cout << "Loaded " << images.size() << " synthetic images" << std::endl;
        return true;
    }
    
    std::pair<torch::Tensor, torch::Tensor> get_batch(int start_idx, int batch_size) {
        int actual_batch_size = std::min(batch_size, (int)images.size() - start_idx);
        
        auto batch_images = torch::zeros({actual_batch_size, 784});
        auto batch_labels = torch::zeros({actual_batch_size}, torch::kLong);
        
        for (int i = 0; i < actual_batch_size; ++i) {
            for (int j = 0; j < 784; ++j) {
                batch_images[i][j] = images[start_idx + i][j];
            }
            batch_labels[i] = labels[start_idx + i];
        }
        
        return {batch_images, batch_labels};
    }
    
    int size() const { return images.size(); }
};

int main() {
    std::cout << "=== MNIST Digit Classification (10 Classes) ===" << std::endl;
    
    // Check CUDA
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    // Load data
    SimpleMNIST train_data, test_data;
    train_data.load_data("", "", 1000);  // Reduce to 1000 training samples
    test_data.load_data("", "", 200);   // Reduce to 200 test samples
    
    // Network architecture: 784 -> 128 -> 64 -> 10
    auto w1 = torch::randn({784, 128}, torch::TensorOptions().device(device).requires_grad(true));
    auto b1 = torch::zeros({128}, torch::TensorOptions().device(device).requires_grad(true));
    auto w2 = torch::randn({128, 64}, torch::TensorOptions().device(device).requires_grad(true));
    auto b2 = torch::zeros({64}, torch::TensorOptions().device(device).requires_grad(true));
    auto w3 = torch::randn({64, 10}, torch::TensorOptions().device(device).requires_grad(true));
    auto b3 = torch::zeros({10}, torch::TensorOptions().device(device).requires_grad(true));
    
    // Xavier initialization
    {
        torch::NoGradGuard no_grad;
        torch::nn::init::xavier_uniform_(w1);
        torch::nn::init::xavier_uniform_(w2);
        torch::nn::init::xavier_uniform_(w3);
    }
    
    float learning_rate = 0.01f;  // Increase from 0.001 to 0.01
    int batch_size = 100;
    int epochs = 20;  // Reduce from 50 to 20
    
    std::cout << "Network: 784 -> 128 -> 64 -> 10" << std::endl;
    std::cout << "Training samples: " << train_data.size() << std::endl;
    std::cout << "Test samples: " << test_data.size() << std::endl;
    std::cout << "Starting training..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Mini-batch training
        for (int start_idx = 0; start_idx < train_data.size(); start_idx += batch_size) {
            auto [batch_images, batch_labels] = train_data.get_batch(start_idx, batch_size);
            batch_images = batch_images.to(device);
            batch_labels = batch_labels.to(device);
            
            // Forward pass
            auto z1 = torch::mm(batch_images, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            
            // Cross-entropy loss
            auto log_probs = torch::log_softmax(z3, 1);
            auto loss = torch::nll_loss(log_probs, batch_labels);
            
            // Backward pass
            loss.backward();
            
            // Update weights
            {
                torch::NoGradGuard no_grad;
                w1 -= learning_rate * w1.grad();
                b1 -= learning_rate * b1.grad();
                w2 -= learning_rate * w2.grad();
                b2 -= learning_rate * b2.grad();
                w3 -= learning_rate * w3.grad();
                b3 -= learning_rate * b3.grad();
                
                // Zero gradients
                w1.grad().zero_();
                b1.grad().zero_();
                w2.grad().zero_();
                b2.grad().zero_();
                w3.grad().zero_();
                b3.grad().zero_();
            }
            
            total_loss += loss.item<float>();
            num_batches++;
        }
        
        if ((epoch + 1) % 5 == 0) {  // Print every 5 epochs instead of 10
            std::cout << "Epoch " << (epoch + 1) << ", Average Loss: " << (total_loss / num_batches) << std::endl;
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    // Test accuracy
    {
        torch::NoGradGuard no_grad;
        auto [test_images, test_labels] = test_data.get_batch(0, test_data.size());
        test_images = test_images.to(device);
        test_labels = test_labels.to(device);
        
        auto z1 = torch::mm(test_images, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto a2 = torch::relu(z2);
        auto z3 = torch::mm(a2, w3) + b3;
        auto predictions = torch::argmax(z3, 1);
        
        auto correct = torch::sum(predictions == test_labels).item<int>();
        auto accuracy = (float)correct / test_data.size() * 100.0f;
        
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
    }
    
    // Test individual digits
    std::cout << "\nTesting individual digits:" << std::endl;
    for (int digit = 0; digit < 10; ++digit) {
        auto [test_image, test_label] = test_data.get_batch(digit * 100, 1);
        test_image = test_image.to(device);
        
        torch::NoGradGuard no_grad;
        auto z1 = torch::mm(test_image, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto a2 = torch::relu(z2);
        auto z3 = torch::mm(a2, w3) + b3;
        auto probs = torch::softmax(z3, 1);
        auto prediction = torch::argmax(probs, 1).item<int>();
        
        std::cout << "Expected digit " << digit << " -> Predicted: " << prediction;
        std::cout << " (confidence: " << probs[0][prediction].item<float>() << ")" << std::endl;
    }
    
    return 0;
}
