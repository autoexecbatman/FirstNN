#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>

class ConsoleVisualizer {
private:
    void clear_screen() {
        #ifdef _WIN32
            system("cls");
        #else
            system("clear");
        #endif
    }
    
public:
    void show_digit(const torch::Tensor& digit_tensor, int prediction, float confidence, int actual = -1) {
        std::cout << "\n=== Digit Visualization ===" << std::endl;
        
        // Convert tensor to CPU and get data
        auto digit_cpu = digit_tensor.cpu();
        auto accessor = digit_cpu.accessor<float, 1>();
        
        // Display 28x28 digit as ASCII art
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                float pixel = accessor[row * 28 + col];
                
                if (pixel > 0.7f) std::cout << "██";
                else if (pixel > 0.5f) std::cout << "▓▓";
                else if (pixel > 0.3f) std::cout << "▒▒";
                else if (pixel > 0.1f) std::cout << "░░";
                else std::cout << "  ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nPredicted: " << prediction 
                  << " | Confidence: " << std::fixed << std::setprecision(1) 
                  << (confidence * 100) << "%";
        
        if (actual >= 0) {
            std::cout << " | Actual: " << actual;
            if (actual == prediction) {
                std::cout << " ✓ CORRECT";
            } else {
                std::cout << " ✗ WRONG";
            }
        }
        std::cout << std::endl;
    }
    
    void show_training_progress(int epoch, float loss, float accuracy = -1) {
        // Simple progress bar
        int progress = std::min(50, epoch / 20); // 50 chars max, scale by epoch
        
        std::cout << "\rEpoch " << std::setw(4) << epoch 
                  << " | Loss: " << std::fixed << std::setprecision(4) << loss
                  << " | [";
        
        for (int i = 0; i < 50; ++i) {
            if (i < progress) std::cout << "=";
            else if (i == progress) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] ";
        
        if (accuracy >= 0) {
            std::cout << "Acc: " << std::setprecision(1) << (accuracy * 100) << "%";
        }
        
        std::cout << std::flush;
    }
    
    void show_predictions_grid(const std::vector<torch::Tensor>& digits, 
                              const std::vector<int>& predictions,
                              const std::vector<float>& confidences,
                              const std::vector<int>& actuals = {}) {
        
        std::cout << "\n=== Predictions Grid ===" << std::endl;
        
        int num_digits = std::min((int)digits.size(), 9);
        
        for (int i = 0; i < num_digits; ++i) {
            std::cout << "\n[" << (i + 1) << "] ";
            std::cout << "Pred: " << predictions[i] 
                      << " (" << std::setprecision(0) << (confidences[i] * 100) << "%)";
            
            if (!actuals.empty()) {
                std::cout << " | Actual: " << actuals[i];
                if (actuals[i] == predictions[i]) {
                    std::cout << " ✓";
                } else {
                    std::cout << " ✗";
                }
            }
            std::cout << std::endl;
            
            // Show mini digit (14x14 compressed)
            auto digit_cpu = digits[i].cpu();
            auto accessor = digit_cpu.accessor<float, 1>();
            
            for (int row = 0; row < 14; ++row) {
                std::cout << "   ";
                for (int col = 0; col < 14; ++col) {
                    // Sample every 2nd pixel for compression
                    float pixel = accessor[(row * 2) * 28 + (col * 2)];
                    
                    if (pixel > 0.5f) std::cout << "█";
                    else if (pixel > 0.2f) std::cout << "▒";
                    else std::cout << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    void show_model_info(const std::string& architecture, int total_params) {
        std::cout << "\n╔════════════════════════════════════╗" << std::endl;
        std::cout << "║          MODEL INFORMATION         ║" << std::endl;
        std::cout << "╠════════════════════════════════════╣" << std::endl;
        std::cout << "║ Architecture: " << std::setw(19) << architecture << " ║" << std::endl;
        std::cout << "║ Parameters:   " << std::setw(19) << total_params << " ║" << std::endl;
        std::cout << "╚════════════════════════════════════╝" << std::endl;
    }
    
    void show_training_header() {
        std::cout << "\n┌─────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│                     TRAINING PROGRESS                      │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────┘" << std::endl;
    }
    
    void wait_for_key(const std::string& prompt = "Press Enter to continue...") {
        std::cout << "\n" << prompt;
        std::cin.ignore();
        std::cin.get();
    }
};
