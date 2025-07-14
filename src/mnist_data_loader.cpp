#include "mnist_data_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

MNISTDataLoader::MNISTData MNISTDataLoader::load_train_data(const std::string& csv_path) {
    std::cout << "Loading MNIST training data from: " << csv_path << std::endl;
    return load_csv_data(csv_path);
}

MNISTDataLoader::MNISTData MNISTDataLoader::load_test_data(const std::string& csv_path) {
    std::cout << "Loading MNIST test data from: " << csv_path << std::endl;
    return load_csv_data(csv_path);
}

MNISTDataLoader::MNISTData MNISTDataLoader::load_csv_data(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csv_path);
    }
    
    std::vector<std::vector<float>> image_data;
    std::vector<int> label_data;
    std::string line;
    bool first_line = true;
    
    std::cout << "Reading CSV data..." << std::endl;
    
    while (std::getline(file, line)) {
        if (first_line) {
            // Skip header line
            first_line = false;
            std::cout << "Header: " << line.substr(0, 50) << "..." << std::endl;
            continue;
        }
        
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<float> parsed_line = parse_csv_line(line);
        
        if (parsed_line.empty()) {
            std::cout << "Warning: Empty line encountered" << std::endl;
            continue;
        }
        
        // First column is label, rest are pixel values
        int label = static_cast<int>(parsed_line[0]);
        std::vector<float> pixels(parsed_line.begin() + 1, parsed_line.end());
        
        // Validate data
        if (pixels.size() != 784) {
            std::cout << "Warning: Expected 784 pixels, got " << pixels.size() << std::endl;
            continue;
        }
        
        if (label < 0 || label > 9) {
            std::cout << "Warning: Invalid label " << label << std::endl;
            continue;
        }
        
        // Normalize pixels to [0, 1] range
        for (float& pixel : pixels) {
            pixel = pixel / 255.0f;
        }
        
        label_data.push_back(label);
        image_data.push_back(pixels);
        
        // Progress indicator
        if (label_data.size() % 10000 == 0) {
            std::cout << "Loaded " << label_data.size() << " samples..." << std::endl;
        }
    }
    
    file.close();
    
    if (image_data.empty()) {
        throw std::runtime_error("No valid data found in CSV file");
    }
    
    std::cout << "Loaded " << image_data.size() << " samples successfully" << std::endl;
    
    // Convert to PyTorch tensors
    int num_samples = image_data.size();
    torch::Tensor images = torch::zeros({num_samples, 784}, torch::kFloat);
    torch::Tensor labels = torch::zeros({num_samples}, torch::kLong);
    
    // Fill tensors
    for (int i = 0; i < num_samples; ++i) {
        auto image_tensor = torch::from_blob(image_data[i].data(), {784}, torch::kFloat);
        images[i] = image_tensor.clone();
        labels[i] = label_data[i];
    }
    
    std::cout << "Created tensors - Images: " << images.sizes() << ", Labels: " << labels.sizes() << std::endl;
    
    // Verify data ranges
    auto min_pixel = torch::min(images).item<float>();
    auto max_pixel = torch::max(images).item<float>();
    auto min_label = torch::min(labels).item<int>();
    auto max_label = torch::max(labels).item<int>();
    
    std::cout << "Pixel range: [" << min_pixel << ", " << max_pixel << "]" << std::endl;
    std::cout << "Label range: [" << min_label << ", " << max_label << "]" << std::endl;
    
    return {images, labels, num_samples};
}

std::vector<float> MNISTDataLoader::parse_csv_line(const std::string& line) {
    std::vector<float> values;
    std::stringstream ss(line);
    std::string cell;
    
    while (std::getline(ss, cell, ',')) {
        if (!cell.empty()) {
            try {
                float value = std::stof(cell);
                values.push_back(value);
            } catch (const std::exception& e) {
                std::cout << "Warning: Cannot parse value '" << cell << "': " << e.what() << std::endl;
                return {}; // Return empty vector on parse error
            }
        }
    }
    
    return values;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> MNISTDataLoader::create_batches(
    const torch::Tensor& images, 
    const torch::Tensor& labels, 
    int batch_size) {
    
    std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
    int num_samples = images.size(0);
    
    for (int start_idx = 0; start_idx < num_samples; start_idx += batch_size) {
        int end_idx = std::min(start_idx + batch_size, num_samples);
        
        auto batch_images = images.slice(0, start_idx, end_idx);
        auto batch_labels = labels.slice(0, start_idx, end_idx);
        
        batches.emplace_back(batch_images, batch_labels);
    }
    
    std::cout << "Created " << batches.size() << " batches of size " << batch_size << std::endl;
    return batches;
}
