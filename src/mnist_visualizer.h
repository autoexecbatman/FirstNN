#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

class MNISTVisualizer {
private:
    cv::Mat canvas;
    int window_width = 800;
    int window_height = 600;
    
public:
    MNISTVisualizer();
    
    void show_digit(const torch::Tensor& digit_tensor, int prediction, float confidence, int actual = -1);
    void show_training_progress(int epoch, float loss, float accuracy);
    void show_predictions_grid(const std::vector<torch::Tensor>& digits, 
                              const std::vector<int>& predictions,
                              const std::vector<float>& confidences,
                              const std::vector<int>& actuals = {});
    bool wait_for_key(int delay = 0);
    void close();
};
