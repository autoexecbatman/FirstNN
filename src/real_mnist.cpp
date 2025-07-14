#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "mnist_visualizer.h"
#include "hand_drawing.h"
#include "mnist_data_loader.h"
#include <iomanip>

int main() {
    std::cout << "=== MNIST Digit Classifier with Real MNIST Data ===" << std::endl;
    
    // Initialize OpenCV visualizer
    MNISTVisualizer visualizer;
    
    // Check CUDA
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    // Load real MNIST data
    std::cout << "\n=== Loading Real MNIST Dataset ===" << std::endl;
    MNISTDataLoader::MNISTData train_data, test_data;
    
    try {
        train_data = MNISTDataLoader::load_train_data();
        test_data = MNISTDataLoader::load_test_data();
    } catch (const std::exception& e) {
        std::cout << "Error loading MNIST data: " << e.what() << std::endl;
        std::cout << "\nPlease run the Python script to convert parquet files to CSV:" << std::endl;
        std::cout << "python convert_parquet_to_csv.py" << std::endl;
        return -1;
    }
    
    // Move to device
    auto train_images = train_data.images.to(device);
    auto train_labels = train_data.labels.to(device);
    auto test_images = test_data.images.to(device);
    auto test_labels = test_data.labels.to(device);
    
    std::cout << "Training data shape: " << train_images.sizes() << std::endl;
    std::cout << "Training labels shape: " << train_labels.sizes() << std::endl;
    std::cout << "Test data shape: " << test_images.sizes() << std::endl;
    std::cout << "Test labels shape: " << test_labels.sizes() << std::endl;
    
    // Show some sample data statistics
    {
        auto label_counts = torch::bincount(train_labels);
        std::cout << "Training label distribution: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << "(" << i << ":" << label_counts[i].item<int>() << ") ";
        }
        std::cout << std::endl;
    }
    
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
    
    float learning_rate = 0.001f;
    int batch_size = 200;
    int epochs = 2000;  // Reduced epochs - real data trains faster
    
    std::cout << "\nNetwork: 784 -> 256 -> 128 -> 10" << std::endl;
    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
    
    // Create validation split (use last 10% of training data)
    int val_split = train_images.size(0) * 0.9;
    auto val_images = train_images.slice(0, val_split, train_images.size(0));
    auto val_labels = train_labels.slice(0, val_split, train_labels.size(0));
    train_images = train_images.slice(0, 0, val_split);
    train_labels = train_labels.slice(0, 0, val_split);
    
    std::cout << "Training samples: " << train_images.size(0) << std::endl;
    std::cout << "Validation samples: " << val_images.size(0) << std::endl;
    
    // Training loop with validation monitoring
    float best_val_accuracy = 0.0f;
    int patience_counter = 0;
    const int patience = 200;  // Early stopping patience
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Training
        for (int start_idx = 0; start_idx < train_images.size(0); start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, (int)train_images.size(0));
            
            auto batch_images = train_images.slice(0, start_idx, end_idx);
            auto batch_labels = train_labels.slice(0, start_idx, end_idx);
            
            // Forward pass (no dropout - real data is more diverse)
            auto z1 = torch::mm(batch_images, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            
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
        
        // Validation evaluation every epoch
        float val_accuracy = 0.0f;
        {
            torch::NoGradGuard no_grad;
            auto z1 = torch::mm(val_images, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            auto predictions = torch::argmax(z3, 1);
            
            auto correct = torch::sum(predictions == val_labels).item<int>();
            val_accuracy = (float)correct / val_images.size(0) * 100.0f;
        }
        
        float avg_loss = total_loss / num_batches;
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << ", Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Val Accuracy: " << std::fixed << std::setprecision(2) << val_accuracy << "%" << std::endl;
        
        // Show training progress visualization
        visualizer.show_training_progress(epoch + 1, avg_loss, val_accuracy);
        
        // Early stopping check
        if (val_accuracy > best_val_accuracy) {
            best_val_accuracy = val_accuracy;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                std::cout << "Early stopping triggered. Best validation accuracy: " 
                          << std::fixed << std::setprecision(2) << best_val_accuracy << "%" << std::endl;
                break;
            }
        }
        
        // Test sample every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            torch::NoGradGuard no_grad;
            
            // Get a random test sample
            int sample_idx = rand() % test_images.size(0);
            auto sample_image = test_images[sample_idx].unsqueeze(0);
            auto actual_label = test_labels[sample_idx].item<int>();
            
            auto z1 = torch::mm(sample_image, w1) + b1;
            auto a1 = torch::relu(z1);
            auto z2 = torch::mm(a1, w2) + b2;
            auto a2 = torch::relu(z2);
            auto z3 = torch::mm(a2, w3) + b3;
            auto probs = torch::softmax(z3, 1);
            auto prediction = torch::argmax(probs, 1).item<int>();
            float confidence = probs[0][prediction].item<float>();
            
            // Show sample visualization
            visualizer.show_digit(test_images[sample_idx], prediction, confidence, actual_label);
            
            std::cout << "Sample test: Actual " << actual_label << " -> Predicted " << prediction 
                      << " (" << (int)(confidence * 100) << "%)" 
                      << (prediction == actual_label ? " CORRECT" : " WRONG") << std::endl;
        }
    }
    
    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Best validation accuracy: " << std::fixed << std::setprecision(2) << best_val_accuracy << "%" << std::endl;
    
    // Final test accuracy
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
        
        std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    }
    
    std::cout << "\nStarting hand drawing interface..." << std::endl;
    
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
