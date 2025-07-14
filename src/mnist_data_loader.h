#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class MNISTDataLoader {
public:
    struct MNISTData {
        torch::Tensor images;
        torch::Tensor labels;
        int num_samples;
    };
    
    // Load MNIST data from CSV files
    static MNISTData load_train_data(const std::string& csv_path = "D:/repo/firstNN/data/mnist/train.csv");
    static MNISTData load_test_data(const std::string& csv_path = "D:/repo/firstNN/data/mnist/test.csv");
    
    // Create data batches
    static std::vector<std::pair<torch::Tensor, torch::Tensor>> create_batches(
        const torch::Tensor& images, 
        const torch::Tensor& labels, 
        int batch_size
    );
    
private:
    static MNISTData load_csv_data(const std::string& csv_path);
    static std::vector<float> parse_csv_line(const std::string& line);
};
