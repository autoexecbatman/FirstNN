#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>

// Create realistic MNIST-like synthetic data that actually learns
class ImprovedSyntheticMNIST {
public:
    std::pair<torch::Tensor, torch::Tensor> load_data(int num_samples) {
        std::cout << "Creating improved synthetic MNIST-like data..." << std::endl;
        
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        
        for (int i = 0; i < num_samples; ++i) {
            int digit = i % 10;
            std::vector<float> image(784, 0.0f);
            
            // Create more realistic digit patterns
            create_digit_pattern(image, digit);
            
            images.push_back(image);
            labels.push_back(digit);
        }
        
        // Convert to tensors
        auto image_tensor = torch::zeros({num_samples, 784});
        auto label_tensor = torch::zeros({num_samples}, torch::kLong);
        
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < 784; ++j) {
                image_tensor[i][j] = images[i][j];
            }
            label_tensor[i] = labels[i];
        }
        
        std::cout << "Generated " << num_samples << " synthetic samples with realistic patterns" << std::endl;
        return {image_tensor, label_tensor};
    }
    
private:
    void create_digit_pattern(std::vector<float>& image, int digit) {
        // Create 28x28 patterns that are learnable
        switch(digit) {
            case 0: create_zero(image); break;
            case 1: create_one(image); break;
            case 2: create_two(image); break;
            case 3: create_three(image); break;
            case 4: create_four(image); break;
            case 5: create_five(image); break;
            case 6: create_six(image); break;
            case 7: create_seven(image); break;
            case 8: create_eight(image); break;
            case 9: create_nine(image); break;
        }
        
        // Add some noise
        for (int i = 0; i < 784; ++i) {
            if (image[i] > 0.5f) {
                image[i] = std::min(1.0f, image[i] + (rand() % 20 - 10) / 100.0f);
            }
        }
    }
    
    void create_zero(std::vector<float>& img) {
        // Oval shape
        for (int row = 6; row < 22; ++row) {
            for (int col = 8; col < 20; ++col) {
                int idx = row * 28 + col;
                if ((row == 6 || row == 21) && col >= 10 && col <= 17) img[idx] = 0.8f;
                if ((col == 8 || col == 19) && row >= 8 && row <= 19) img[idx] = 0.8f;
                if ((row >= 8 && row <= 19) && (col == 9 || col == 18)) img[idx] = 0.9f;
            }
        }
    }
    
    void create_one(std::vector<float>& img) {
        // Vertical line
        for (int row = 4; row < 24; ++row) {
            img[row * 28 + 14] = 0.9f;
            if (row < 8) img[row * 28 + 13] = 0.7f;
        }
        // Base
        for (int col = 10; col < 19; ++col) {
            img[23 * 28 + col] = 0.8f;
        }
    }
    
    void create_two(std::vector<float>& img) {
        // Top curve
        for (int col = 8; col < 18; ++col) {
            img[6 * 28 + col] = 0.8f;
            img[7 * 28 + col] = 0.7f;
        }
        // Right vertical
        for (int row = 8; row < 14; ++row) {
            img[row * 28 + 17] = 0.8f;
        }
        // Diagonal
        for (int i = 0; i < 8; ++i) {
            img[(14 + i) * 28 + (17 - i * 2)] = 0.8f;
        }
        // Bottom line
        for (int col = 7; col < 20; ++col) {
            img[22 * 28 + col] = 0.9f;
        }
    }
    
    void create_three(std::vector<float>& img) {
        // Top horizontal
        for (int col = 8; col < 18; ++col) {
            img[6 * 28 + col] = 0.8f;
        }
        // Middle horizontal
        for (int col = 10; col < 17; ++col) {
            img[14 * 28 + col] = 0.8f;
        }
        // Bottom horizontal
        for (int col = 8; col < 18; ++col) {
            img[22 * 28 + col] = 0.8f;
        }
        // Right verticals
        for (int row = 7; row < 13; ++row) {
            img[row * 28 + 17] = 0.8f;
        }
        for (int row = 15; row < 22; ++row) {
            img[row * 28 + 17] = 0.8f;
        }
    }
    
    void create_four(std::vector<float>& img) {
        // Left vertical
        for (int row = 4; row < 16; ++row) {
            img[row * 28 + 9] = 0.8f;
        }
        // Right vertical
        for (int row = 4; row < 24; ++row) {
            img[row * 28 + 17] = 0.9f;
        }
        // Horizontal
        for (int col = 9; col < 18; ++col) {
            img[15 * 28 + col] = 0.8f;
        }
    }
    
    void create_five(std::vector<float>& img) {
        // Top horizontal
        for (int col = 8; col < 18; ++col) {
            img[6 * 28 + col] = 0.8f;
        }
        // Left vertical top
        for (int row = 7; row < 14; ++row) {
            img[row * 28 + 8] = 0.8f;
        }
        // Middle horizontal
        for (int col = 8; col < 17; ++col) {
            img[14 * 28 + col] = 0.8f;
        }
        // Right vertical bottom
        for (int row = 15; row < 22; ++row) {
            img[row * 28 + 17] = 0.8f;
        }
        // Bottom horizontal
        for (int col = 8; col < 18; ++col) {
            img[22 * 28 + col] = 0.8f;
        }
    }
    
    void create_six(std::vector<float>& img) {
        // Similar to 5 but with left curve at bottom
        create_five(img);
        // Add left bottom curve
        for (int row = 15; row < 22; ++row) {
            img[row * 28 + 8] = 0.8f;
        }
    }
    
    void create_seven(std::vector<float>& img) {
        // Top horizontal
        for (int col = 8; col < 19; ++col) {
            img[6 * 28 + col] = 0.9f;
        }
        // Diagonal down
        for (int i = 0; i < 16; ++i) {
            img[(7 + i) * 28 + (18 - i/2)] = 0.8f;
        }
    }
    
    void create_eight(std::vector<float>& img) {
        // Two circles
        create_zero(img);
        // Middle horizontal
        for (int col = 10; col < 17; ++col) {
            img[14 * 28 + col] = 0.8f;
        }
    }
    
    void create_nine(std::vector<float>& img) {
        // Similar to 6 but flipped
        create_six(img);
        // Remove left bottom, add right top
        for (int row = 15; row < 22; ++row) {
            img[row * 28 + 8] = 0.0f;
        }
        for (int row = 7; row < 14; ++row) {
            img[row * 28 + 17] = 0.8f;
        }
    }
};

int main() {
    std::cout << "=== Improved Synthetic MNIST Training ===" << std::endl;
    
    // Check CUDA
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    // Generate improved synthetic data
    ImprovedSyntheticMNIST generator;
    auto [train_images, train_labels] = generator.load_data(10000);  // More data
    auto [test_images, test_labels] = generator.load_data(2000);
    
    // Move to device
    train_images = train_images.to(device);
    train_labels = train_labels.to(device);
    test_images = test_images.to(device);
    test_labels = test_labels.to(device);
    
    std::cout << "Training data shape: " << train_images.sizes() << std::endl;
    std::cout << "Training labels shape: " << train_labels.sizes() << std::endl;
    
    // Print first few labels
    std::cout << "First 10 labels: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << train_labels[i].item<int>() << " ";
    }
    std::cout << std::endl;
    
    // Network architecture: 784 -> 256 -> 128 -> 10 (bigger network)
    auto w1 = torch::randn({784, 256}, torch::TensorOptions().device(device).requires_grad(true));
    auto b1 = torch::zeros({256}, torch::TensorOptions().device(device).requires_grad(true));
    auto w2 = torch::randn({256, 128}, torch::TensorOptions().device(device).requires_grad(true));
    auto b2 = torch::zeros({128}, torch::TensorOptions().device(device).requires_grad(true));
    auto w3 = torch::randn({128, 10}, torch::TensorOptions().device(device).requires_grad(true));
    auto b3 = torch::zeros({10}, torch::TensorOptions().device(device).requires_grad(true));
    
    // Xavier initialization
    {
        torch::NoGradGuard no_grad;
        torch::nn::init::xavier_uniform_(w1);
        torch::nn::init::xavier_uniform_(w2);
        torch::nn::init::xavier_uniform_(w3);
    }
    
    float learning_rate = 0.01f;
    int batch_size = 200;
    int epochs = 5000;  // Reasonable amount
    
    std::cout << "Network: 784 -> 256 -> 128 -> 10" << std::endl;
    std::cout << "Training samples: " << train_images.size(0) << std::endl;
    std::cout << "Test samples: " << test_images.size(0) << std::endl;
    std::cout << "Starting training..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Mini-batch training
        for (int start_idx = 0; start_idx < train_images.size(0); start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, (int)train_images.size(0));
            
            auto batch_images = train_images.slice(0, start_idx, end_idx);
            auto batch_labels = train_labels.slice(0, start_idx, end_idx);
            
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
        
        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch " << (epoch + 1) << ", Average Loss: " << (total_loss / num_batches) << std::endl;
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    // Test accuracy
    {
        torch::NoGradGuard no_grad;
        auto z1 = torch::mm(test_images, w1) + b1;
        auto a1 = torch::relu(z1);
        auto z2 = torch::mm(a1, w2) + b2;
        auto a2 = torch::relu(z2);
        auto z3 = torch::mm(a2, w3) + b3;
        auto predictions = torch::argmax(z3, 1);
        
        auto correct = torch::sum(predictions == test_labels).item<int>();
        auto accuracy = (float)correct / test_images.size(0) * 100.0f;
        
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
    }
    
    // Test each digit class
    std::cout << "\nTesting each digit (0-9):" << std::endl;
    {
        torch::NoGradGuard no_grad;
        for (int digit = 0; digit < 10; ++digit) {
            auto test_image = test_images[digit * 200].unsqueeze(0);  // Take one of each digit
            auto z1 = torch::mm(test_image, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            auto probs = torch::softmax(z3, 1);
            auto prediction = torch::argmax(probs, 1).item<int>();
            
            std::cout << "Digit " << digit << " -> Predicted: " << prediction 
                      << ", Confidence: " << probs[0][prediction].item<float>() << std::endl;
        }
    }
    
    return 0;
}
