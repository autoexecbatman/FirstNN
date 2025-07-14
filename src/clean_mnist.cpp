#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "mnist_visualizer.h"
#include "hand_drawing.h"

// Simple MNIST-like data generator with manual testing
class MNISTGenerator {
public:
    std::pair<torch::Tensor, torch::Tensor> load_data(int num_samples) {
        std::cout << "Creating realistic MNIST-like data..." << std::endl;
        
        auto images = torch::zeros({num_samples, 784});
        auto labels = torch::zeros({num_samples}, torch::kLong);
        
        for (int i = 0; i < num_samples; ++i) {
            int digit = i % 10;
            labels[i] = digit;
            
            // Create realistic digit pattern
            auto image_view = images[i];
            create_realistic_digit(image_view, digit);
        }
        
        std::cout << "Generated " << num_samples << " realistic MNIST-like samples" << std::endl;
        return {images, labels};
    }
    
    void create_test_digit(torch::Tensor& img, int digit) {
        create_realistic_digit(img, digit);
    }
    
private:
    void create_realistic_digit(torch::Tensor& img, int digit) {
        img.fill_(0.0f);
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
                float noise = (rand() % 40 - 20) / 200.0f;
                data[i] = std::max(0.0f, std::min(1.0f, data[i] + noise));
            }
        }
    }
    
    void set_pixel(float* data, int row, int col, float value) {
        if (row >= 0 && row < 28 && col >= 0 && col < 28) {
            data[row * 28 + col] = value;
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
    
    void create_zero_pattern(float* data) {
        for (int row = 6; row < 22; ++row) {
            for (int col = 8; col < 20; ++col) {
                float dist = sqrt((row - 14) * (row - 14) + (col - 14) * (col - 14));
                if (dist > 4 && dist < 8) {
                    set_pixel(data, row, col, 0.8f);
                }
            }
        }
    }
    
    void create_one_pattern(float* data) {
        for (int row = 4; row < 24; ++row) {
            set_pixel(data, row, 14, 0.9f);
        }
        for (int col = 10; col < 19; ++col) {
            set_pixel(data, 23, col, 0.7f);
        }
    }
    
    void create_two_pattern(float* data) {
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 6, col, 0.8f);
        }
        for (int row = 8; row < 14; ++row) {
            set_pixel(data, row, 17, 0.8f);
        }
        for (int i = 0; i < 8; ++i) {
            set_pixel(data, 14 + i, 17 - i * 2, 0.8f);
        }
        for (int col = 7; col < 20; ++col) {
            set_pixel(data, 22, col, 0.9f);
        }
    }
    
    void create_four_pattern(float* data) {
        for (int row = 4; row < 16; ++row) {
            set_pixel(data, row, 9, 0.8f);
        }
        for (int row = 4; row < 24; ++row) {
            set_pixel(data, row, 17, 0.9f);
        }
        for (int col = 9; col < 18; ++col) {
            set_pixel(data, 15, col, 0.8f);
        }
    }
    
    void create_five_pattern(float* data) {
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 6, col, 0.8f);
        }
        for (int row = 7; row < 14; ++row) {
            set_pixel(data, row, 8, 0.8f);
        }
        for (int col = 8; col < 17; ++col) {
            set_pixel(data, 14, col, 0.8f);
        }
        for (int row = 15; row < 22; ++row) {
            set_pixel(data, row, 17, 0.8f);
        }
        for (int col = 8; col < 18; ++col) {
            set_pixel(data, 22, col, 0.8f);
        }
    }
    
    void create_six_pattern(float* data) {
        create_five_pattern(data);
        for (int row = 15; row < 22; ++row) {
            set_pixel(data, row, 8, 0.8f);
        }
    }
    
    void create_seven_pattern(float* data) {
        for (int col = 8; col < 19; ++col) {
            set_pixel(data, 6, col, 0.9f);
        }
        for (int i = 0; i < 16; ++i) {
            set_pixel(data, 8 + i, 18 - i/2, 0.8f);
        }
    }
    
    void create_eight_pattern(float* data) {
        for (int row = 6; row < 15; ++row) {
            for (int col = 9; col < 19; ++col) {
                float dist = sqrt((row - 10.5) * (row - 10.5) + (col - 14) * (col - 14));
                if (dist > 2.5 && dist < 4.5) {
                    set_pixel(data, row, col, 0.8f);
                }
            }
        }
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
        for (int row = 6; row < 15; ++row) {
            for (int col = 9; col < 19; ++col) {
                float dist = sqrt((row - 10) * (row - 10) + (col - 14) * (col - 14));
                if (dist > 2.5 && dist < 4.5) {
                    set_pixel(data, row, col, 0.8f);
                }
            }
        }
        for (int row = 15; row < 23; ++row) {
            set_pixel(data, row, 17, 0.8f);
        }
    }
};

int main() {
    std::cout << "=== MNIST Digit Classifier with OpenCV Visualization ===" << std::endl;
    
    // Initialize OpenCV visualizer
    MNISTVisualizer visualizer;
    
    // Check CUDA
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    // Load data
    MNISTGenerator generator;
    auto [train_images, train_labels] = generator.load_data(5000);
    auto [test_images, test_labels] = generator.load_data(1000);
    
    // Move to device
    train_images = train_images.to(device);
    train_labels = train_labels.to(device);
    test_images = test_images.to(device);
    test_labels = test_labels.to(device);
    
    std::cout << "Training data shape: " << train_images.sizes() << std::endl;
    std::cout << "Training labels shape: " << train_labels.sizes() << std::endl;
    
    // Network: 784 -> 256 -> 128 -> 10
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
    int epochs = 10000;  // Reduced for faster testing
    
    std::cout << "Network: 784 -> 256 -> 128 -> 10" << std::endl;
    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        for (int start_idx = 0; start_idx < train_images.size(0); start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, (int)train_images.size(0));
            
            auto batch_images = train_images.slice(0, start_idx, end_idx);
            auto batch_labels = train_labels.slice(0, start_idx, end_idx);
            
            // Forward pass with dropout
            auto z1 = torch::mm(batch_images, w1) + b1;
            auto a1 = torch::relu(z1);
            auto a1_dropout = torch::dropout(a1, 0.3, true);
            auto z2 = torch::mm(a1_dropout, w2) + b2;
            auto a2 = torch::relu(z2);
            auto a2_dropout = torch::dropout(a2, 0.3, true);
            auto z3 = torch::mm(a2_dropout, w3) + b3;
            
            auto log_probs = torch::log_softmax(z3, 1);
            auto loss = torch::nll_loss(log_probs, batch_labels);
            
            loss.backward();
            
            {
                torch::NoGradGuard no_grad;
                w1 -= learning_rate * w1.grad();
                b1 -= learning_rate * b1.grad();
                w2 -= learning_rate * w2.grad();
                b2 -= learning_rate * b2.grad();
                w3 -= learning_rate * w3.grad();
                b3 -= learning_rate * b3.grad();
                
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
            float avg_loss = total_loss / num_batches;
            std::cout << "Epoch " << (epoch + 1) << ", Average Loss: " << avg_loss << std::endl;
            
            // Show training progress visualization
            visualizer.show_training_progress(epoch + 1, avg_loss, 0.0f);
            
            // Auto-test with sample digits every 100 epochs
            {
                torch::NoGradGuard no_grad;
                
                // Test with digit 3
                auto sample_image = torch::zeros({1, 784});
                auto sample_tensor = sample_image[0];
                generator.create_test_digit(sample_tensor, 3);
                sample_image = sample_image.to(device);
                
                auto z1 = torch::mm(sample_image, w1) + b1;
                auto a1 = torch::relu(z1);
                auto z2 = torch::mm(a1, w2) + b2;
                auto a2 = torch::relu(z2);
                auto z3 = torch::mm(a2, w3) + b3;
                auto probs = torch::softmax(z3, 1);
                auto prediction = torch::argmax(probs, 1).item<int>();
                float confidence = probs[0][prediction].item<float>();
                
                // Show automatic digit visualization
                visualizer.show_digit(sample_tensor, prediction, confidence, 3);
                
                std::cout << "Auto-test: Digit 3 -> Predicted: " << prediction 
                          << ", Confidence: " << (confidence * 100) << "%" << std::endl;
            }
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    
    // Automatic final test with multiple digits
    std::cout << "\nRunning automatic final test..." << std::endl;
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::Tensor> test_digits;
        std::vector<int> predictions;
        std::vector<float> confidences;
        std::vector<int> actuals;
        
        for (int i = 0; i < 9; ++i) {
            int actual_digit = i % 10;
            auto test_img = torch::zeros({784});
            generator.create_test_digit(test_img, actual_digit);
            test_digits.push_back(test_img);
            actuals.push_back(actual_digit);
            
            // Get prediction
            auto img_batch = test_img.unsqueeze(0).to(device);
            auto z1 = torch::mm(img_batch, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            auto probs = torch::softmax(z3, 1);
            auto pred = torch::argmax(probs, 1).item<int>();
            auto conf = probs[0][pred].item<float>();
            
            predictions.push_back(pred);
            confidences.push_back(conf);
            
            std::cout << "Digit " << actual_digit << " -> Predicted: " << pred 
                      << " (" << (int)(conf * 100) << "%)" 
                      << (pred == actual_digit ? " CORRECT" : " WRONG") << std::endl;
        }
        
        // Show automatic grid visualization
        visualizer.show_predictions_grid(test_digits, predictions, confidences, actuals);
        std::cout << "\nAutomatic test complete! Check the visualization windows." << std::endl;
    }
    
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
    
    std::cout << "\nVisualization complete! Starting hand drawing interface..." << std::endl;
    
    // Initialize hand drawing interface
    HandDrawingInterface drawing_interface;
    drawing_interface.set_network_params(w1, b1, w2, b2, w3, b3, device);
    
    // Start interactive drawing session
    drawing_interface.start_drawing_session();
    
    // Wait for user to view results
    visualizer.wait_for_key();
    visualizer.close();
    
    return 0;
}
