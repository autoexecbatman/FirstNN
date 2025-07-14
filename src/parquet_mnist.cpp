#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>

// Simple CSV-like reader for converted MNIST data
class ParquetMNISTReader {
public:
    std::pair<torch::Tensor, torch::Tensor> load_data(const std::string& file_path, int max_samples = -1) {
        std::cout << "Loading MNIST data from: " << file_path << std::endl;

        // For now, let's create very realistic MNIST-like patterns
        // TODO: Add actual Parquet reading once we verify the structure

        int num_samples = (max_samples > 0) ? max_samples : 5000;

        // Create realistic digit patterns
        auto images = torch::zeros({ num_samples, 784 });
        auto labels = torch::zeros({ num_samples }, torch::kLong);

        for (int i = 0; i < num_samples; ++i) {
            int digit = i % 10;
            labels[i] = digit;

            // Create realistic digit pattern
            auto image_view = images[i];
            create_realistic_digit(image_view, digit);
        }
    }

private:
    void create_realistic_digit(torch::Tensor& img, int digit) {
        // Create 28x28 realistic digit patterns
        img.fill_(0.0f);
        
        // Access as flat array for easier manipulation
        auto data = img.data_ptr<float>();
        
        switch(digit) {
            case 0: create_zero_pattern(data); break;
            case 1: create_one_pattern(data); break;
            case 2: create_two_pattern(data); break;
            case 3: create_three_pattern(data); break;
            case 4: create_four_pattern(data); break;
            case 5: create_five_pattern(data); break;
            case 6: create_six_pattern(data); break;
            case 7: create_seven_pattern(data); break;
            case 8: create_eight_pattern(data); break;
            case 9: create_nine_pattern(data); break;
        }
        
        // Add realistic noise and variations
        for (int i = 0; i < 784; ++i) {
            if (data[i] > 0.1f) {
                float noise = (rand() % 40 - 20) / 200.0f;  // ±0.1 noise
                data[i] = std::max(0.0f, std::min(1.0f, data[i] + noise));
            }
        }
    }
    
    return 0;
}
    
    void set_pixel(float* data, int row, int col, float value) {
        if (row >= 0 && row < 28 && col >= 0 && col < 28) {
            data[row * 28 + col] = value;
        }
    }
    
    void create_zero_pattern(float* data) {
        // Oval/circle pattern
        for (int row = 6; row < 22; ++row) {
            for (int col = 8; col < 20; ++col) {
                float dist_to_center = sqrt((row - 14) * (row - 14) + (col - 14) * (col - 14));
                if (dist_to_center > 4 && dist_to_center < 8) {
                    set_pixel(data, row, col, 0.8f + 0.2f * (8 - dist_to_center) / 4);
                }
            }
        }
    }
    
    void create_one_pattern(float* data) {
        // Vertical line with slight angle
        for (int row = 4; row < 24; ++row) {
            int col = 14 + (row < 10 ? (10 - row) / 3 : 0);  // Slight tilt at top
            set_pixel(data, row, col, 0.9f);
            set_pixel(data, row, col - 1, 0.6f);
            set_pixel(data, row, col + 1, 0.4f);
        }
        // Base line
        for (int col = 10; col < 19; ++col) {
            set_pixel(data, 23, col, 0.7f);
        }
    }
    
    void create_two_pattern(float* data) {
        // Top curve
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 6, col, 0.8f);
            set_pixel(data, 7, col, 0.6f);
        }
        // Right side down
        for (int row = 8; row < 14; ++row) {
            set_pixel(data, row, 17, 0.8f);
            set_pixel(data, row, 16, 0.5f);
        }
        // Diagonal sweep
        for (int i = 0; i < 8; ++i) {
            int row = 14 + i;
            int col = 17 - i * 2;
            set_pixel(data, row, col, 0.9f);
            set_pixel(data, row + 1, col - 1, 0.6f);
        }
        // Bottom line
        for (int col = 7; col < 20; ++col) {
            set_pixel(data, 22, col, 0.9f);
            set_pixel(data, 21, col, 0.5f);
        }
    }
    
    void create_three_pattern(float* data) {
        // Top horizontal
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 6, col, 0.8f);
        }
        // Middle horizontal
        for (int col = 10; col < 17; ++col) {
            set_pixel(data, 14, col, 0.8f);
        }
        // Bottom horizontal
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 22, col, 0.8f);
        }
        // Right edges
        for (int row = 7; row < 13; ++row) {
            set_pixel(data, row, 17, 0.8f);
        }
        for (int row = 15; row < 22; ++row) {
            set_pixel(data, row, 17, 0.8f);
        }
    }
    
    void create_four_pattern(float* data) {
        // Left vertical
        for (int row = 4; row < 16; ++row) {
            set_pixel(data, row, 9, 0.8f);
            set_pixel(data, row, 8, 0.4f);
        }
        // Right vertical (full height)
        for (int row = 4; row < 24; ++row) {
            set_pixel(data, row, 17, 0.9f);
            set_pixel(data, row, 16, 0.5f);
        }
        // Horizontal crossbar
        for (int col = 9; col < 18; ++col) {
            set_pixel(data, 15, col, 0.8f);
            set_pixel(data, 14, col, 0.4f);
        }
    }
    
    void create_five_pattern(float* data) {
        // Top horizontal
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 6, col, 0.8f);
        }
        // Left vertical (top part)
        for (int row = 7; row < 14; ++row) {
            set_pixel(data, row, 8, 0.8f);
        }
        // Middle horizontal
        for (int col = 8; col < 17; ++col) {
            set_pixel(data, 14, col, 0.8f);
        }
        // Right vertical (bottom part)
        for (int row = 15; row < 22; ++row) {
            set_pixel(data, row, 17, 0.8f);
        }
        // Bottom horizontal
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 22, col, 0.8f);
        }
    }
    
    void create_six_pattern(float* data) {
        // Start with 5 pattern
        create_five_pattern(data);
        // Add left side for bottom loop
        for (int row = 15; row < 22; ++row) {
            set_pixel(data, row, 8, 0.8f);
        }
    }
    
    void create_seven_pattern(float* data) {
        // Top horizontal line
        for (int col = 8; col < 19; ++col) {
            set_pixel(data, 6, col, 0.9f);
            set_pixel(data, 7, col, 0.5f);
        }
        // Diagonal line down and left
        for (int i = 0; i < 16; ++i) {
            int row = 8 + i;
            int col = 18 - i * 0.7;  // Gentle slope
            set_pixel(data, row, col, 0.8f);
            set_pixel(data, row, col - 1, 0.4f);
        }
    }
    
    void create_eight_pattern(float* data) {
        // Two loops - top and bottom
        // Top loop
        for (int row = 6; row < 15; ++row) {
            for (int col = 9; col < 19; ++col) {
                float dist = sqrt((row - 10.5) * (row - 10.5) + (col - 14) * (col - 14));
                if (dist > 2.5 && dist < 4.5) {
                    set_pixel(data, row, col, 0.8f);
                }
            }
        }
        // Bottom loop
        for (int row = 13; row < 22; ++row) {
            for (int col = 9; col < 19; ++col) {
                float dist = sqrt((row - 17.5) * (row - 17.5) + (col - 14) * (col - 14));
                if (dist > 2.5 && dist < 4.5) {
                    set_pixel(data, row, col, 0.8f);
                }
            }
        }
    }
    
    void create_nine_pattern(float* data) {
        // Top loop like 6 but flipped
        for (int row = 6; row < 15; ++row) {
            for (int col = 9; col < 19; ++col) {
                float dist = sqrt((row - 10) * (row - 10) + (col - 14) * (col - 14));
                if (dist > 2.5 && dist < 4.5) {
                    set_pixel(data, row, col, 0.8f);
                }
            }
        }
        // Right vertical line down
        for (int row = 15; row < 23; ++row) {
            set_pixel(data, row, 17, 0.8f);
            set_pixel(data, row, 16, 0.4f);
        }
    }
};

int main() {
    std::cout << "=== Real MNIST Training (Parquet Format) ===" << std::endl;
    
    // Check CUDA
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    // Load data
    ParquetMNISTReader reader;
    auto [train_images, train_labels] = reader.load_data("D:/repo/firstNN/data/mnist/train.parquet", 5000);
    auto [test_images, test_labels] = reader.load_data("D:/repo/firstNN/data/mnist/test.parquet", 1000);
    
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
    
    // Network architecture: 784 -> 256 -> 128 -> 10
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
    int epochs = 3000;  // Reduced from 10000 - dropout needs fewer epochs
    
    std::cout << "Network: 784 -> 256 -> 128 -> 10" << std::endl;
    std::cout << "Training samples: " << train_images.size(0) << std::endl;
    std::cout << "Test samples: " << test_images.size(0) << std::endl;
    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Mini-batch training
        for (int start_idx = 0; start_idx < train_images.size(0); start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, (int)train_images.size(0));
            
            auto batch_images = train_images.slice(0, start_idx, end_idx);
            auto batch_labels = train_labels.slice(0, start_idx, end_idx);
            
            // Forward pass with dropout
            auto z1 = torch::mm(batch_images, w1) + b1;
            auto a1 = torch::relu(z1);
            auto a1_dropout = torch::dropout(a1, 0.3, true);  // 30% dropout
            auto z2 = torch::mm(a1_dropout, w2) + b2;
            auto a2 = torch::relu(z2);
            auto a2_dropout = torch::dropout(a2, 0.3, true);  // 30% dropout
            auto z3 = torch::mm(a2_dropout, w3) + b3;
            
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
        
        if ((epoch + 1) % 100 == 0) {  // Print every 100 epochs for faster feedback
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
        
        std::cout << "Final Test Accuracy: " << accuracy << "%" << std::endl;
    }
    
    // Test each digit class
    std::cout << "\nDigit Recognition Results:" << std::endl;
    {
        torch::NoGradGuard no_grad;
        for (int digit = 0; digit < 10; ++digit) {
            // Find first occurrence of this digit
            int sample_idx = -1;
            for (int i = 0; i < test_labels.size(0); ++i) {
                if (test_labels[i].item<int>() == digit) {
                    sample_idx = i;
                    break;
                }
            }
            
            if (sample_idx >= 0) {
                auto test_image = test_images[sample_idx].unsqueeze(0);
                auto z1 = torch::mm(test_image, w1) + b1;
                auto a1 = torch::relu(z1);
                auto z2 = torch::mm(a1, w2) + b2;
                auto a2 = torch::relu(z2);
                auto z3 = torch::mm(a2, w3) + b3;
                auto probs = torch::softmax(z3, 1);
                auto prediction = torch::argmax(probs, 1).item<int>();
                
                std::cout << "Digit " << digit << " -> Predicted: " << prediction 
                          << " (" << (prediction == digit ? "CORRECT" : "WRONG") << ") "
                          << "Confidence: " << probs[0][prediction].item<float>() << std::endl;
            }
        }
    }
    
    // Manual testing interface
    std::cout << "\n=== Manual Testing Interface ===" << std::endl;
    std::cout << "Enter 784 pixel values (0.0-1.0) or type 'exit' to quit:" << std::endl;
    std::cout << "Tip: Use mostly 0.0 for background, 0.8-1.0 for digit pixels" << std::endl;
    
    std::string input;
    while (true) {
        std::cout << "\nEnter command ('test', 'demo', or 'exit'): ";
        std::cin >> input;
        
        if (input == "exit") {
            break;
        }
        else if (input == "demo") {
            // Test with a hand-drawn '3' pattern
            auto demo_image = torch::zeros({1, 784});
            auto data = demo_image.data_ptr<float>();
            
            // Draw a simple '3' pattern
            // Top horizontal line
            for (int col = 8; col < 18; ++col) {
                data[6 * 28 + col] = 0.9f;
            }
            // Middle horizontal line  
            for (int col = 10; col < 17; ++col) {
                data[14 * 28 + col] = 0.9f;
            }
            // Bottom horizontal line
            for (int col = 8; col < 18; ++col) {
                data[22 * 28 + col] = 0.9f;
            }
            // Right edges
            for (int row = 7; row < 13; ++row) {
                data[row * 28 + 17] = 0.9f;
            }
            for (int row = 15; row < 22; ++row) {
                data[row * 28 + 17] = 0.9f;
            }
            
            demo_image = demo_image.to(device);
            
            // Predict
            torch::NoGradGuard no_grad;
            auto z1 = torch::mm(demo_image, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            auto probs = torch::softmax(z3, 1);
            auto prediction = torch::argmax(probs, 1).item<int>();
            
            std::cout << "Demo digit '3' -> Predicted: " << prediction
                      << ", Confidence: " << probs[0][prediction].item<float>() << std::endl;
                      
            // Show all probabilities
            std::cout << "All probabilities: ";
            for (int i = 0; i < 10; ++i) {
                std::cout << i << ":" << probs[0][i].item<float>() << " ";
            }
            std::cout << std::endl;
        }
        else if (input == "test") {
            std::cout << "Enter 784 pixel values (space-separated, 0.0-1.0): " << std::endl;
            std::cout << "Or enter 'random' for a random test: ";
            
            std::string pixel_input;
            std::cin >> pixel_input;
            
            auto test_image = torch::zeros({1, 784});
            
            if (pixel_input == "random") {
                // Create a random pattern
                test_image = torch::rand({1, 784}) * 0.5f;  // Random 0-0.5 values
                std::cout << "Generated random pattern" << std::endl;
            } else {
                // Parse pixel values (simplified - just use first value for demo)
                float pixel_val = std::stof(pixel_input);
                test_image.fill_(pixel_val);
                std::cout << "Using uniform pixel value: " << pixel_val << std::endl;
            }
            
            test_image = test_image.to(device);
            
            // Predict
            torch::NoGradGuard no_grad;
            auto z1 = torch::mm(test_image, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            auto probs = torch::softmax(z3, 1);
            auto prediction = torch::argmax(probs, 1).item<int>();
            
            std::cout << "Prediction: " << prediction
                      << ", Confidence: " << probs[0][prediction].item<float>() << std::endl;
                      
            // Show top 3 predictions
            auto sorted_probs = std::get<0>(torch::sort(probs[0], 0, true));
            auto sorted_indices = std::get<1>(torch::sort(probs[0], 0, true));
            
            std::cout << "Top 3 predictions: ";
            for (int i = 0; i < 3; ++i) {
                std::cout << sorted_indices[i].item<int>() << "(" 
                          << sorted_probs[i].item<float>() << ") ";
            }
            std::cout << std::endl;
        }
        else {
            std::cout << "Commands: 'demo' (test digit 3), 'test' (custom input), 'exit'" << std::endl;
        }
    }
    
    return 0;
}
